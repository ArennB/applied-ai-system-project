"""Tests for the planning agent in agent.

No real model is involved: FakeLLM replays a scripted list of tool calls, which
makes the agent loop, the guardrails and the post-check fully deterministic.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from agent import (
    AgentResult,
    ModelTurn,
    PlannerTools,
    PlanningAgent,
    ToolCall,
)
from scheduler import Pet, Priority, Task

DAY = date(2026, 10, 7)


def call(tool: str, **args) -> ModelTurn:
    """A model turn making a single tool call."""
    return ModelTurn(tool_calls=[ToolCall(tool, args, id=tool)])


class FakeLLM:
    """Replays scripted turns and records what the agent sent back each time."""

    def __init__(self, turns: list[ModelTurn]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def next_step(self, system, messages, tools):
        self.seen.append(list(messages))
        if not self.turns:
            # Loop forever on a harmless call, to exercise the step cap.
            return call("build_plan")
        return self.turns.pop(0)

    def tool_results(self) -> list[dict]:
        """Every tool result the agent fed back, in order."""
        results = []
        for message in self.seen[-1]:
            if message["role"] == "tool":
                results.extend(result for _, result in message["results"])
        return results


def run(turns, pets=None, minutes=60, max_steps=15) -> tuple[AgentResult, FakeLLM]:
    llm = FakeLLM(turns)
    agent = PlanningAgent(llm, max_steps=max_steps)
    result = agent.run("plan my day", "Jordan", minutes, pets, on=DAY)
    return result, llm


def tools(pets=None, minutes=60) -> PlannerTools:
    return PlannerTools("Jordan", minutes, pets, on=DAY)


# --- Tool guardrails ---------------------------------------------------------


@pytest.mark.parametrize(
    "args, message",
    [
        ({"priority": "urgent"}, "priority must be low, medium or high"),
        ({"duration_minutes": 0}, "between 1 and 240"),
        ({"duration_minutes": 500}, "between 1 and 240"),
        ({"duration_minutes": 12.5}, "whole number"),
        ({"duration_minutes": "twenty"}, "whole number"),
        ({"frequency": "hourly"}, "frequency must be one of"),
        ({"fixed_start": "25:00"}, "HH:MM"),
        ({"pet_name": "Ghost"}, "No pet named 'Ghost'"),
    ],
)
def test_add_task_rejects_bad_input_with_a_readable_error(args, message):
    planner = tools([Pet("Mochi", "dog")])
    good = dict(pet_name="Mochi", title="Walk", duration_minutes=30, priority="high", frequency="daily")
    result = planner.dispatch("add_task", {**good, **args})
    assert message in result["error"]
    assert planner.pets[0].tasks == []


def test_add_task_accepts_loose_but_valid_input():
    planner = tools([Pet("Mochi", "dog")])
    result = planner.dispatch(
        "add_task",
        dict(pet_name="mochi", title=" Vet ", duration_minutes=45.0, priority="HIGH",
             frequency="once", fixed_start="10:00"),
    )
    assert result["ok"]
    task = planner.pets[0].tasks[0]
    assert (task.title, task.duration_minutes, task.priority, task.required_start) == (
        "Vet", 45, Priority.HIGH, 600,
    )
    assert task.due_date == DAY


def test_duplicate_pet_and_bad_species_rejected():
    planner = tools([Pet("Mochi", "dog")])
    assert "already exists" in planner.dispatch("add_pet", {"name": "MOCHI", "animal_type": "dog"})["error"]
    assert "animal_type" in planner.dispatch("add_pet", {"name": "Rex", "animal_type": "dragon"})["error"]


def test_unknown_tool_and_missing_arguments_do_not_raise():
    planner = tools()
    assert "Unknown tool" in planner.dispatch("delete_everything", {})["error"]
    assert "Bad arguments" in planner.dispatch("add_pet", {"name": "Rex"})["error"]


def test_update_task_is_all_or_nothing():
    pet = Pet("Mochi", "dog", [Task("Walk", 30, Priority.HIGH, due_date=DAY)])
    planner = tools([pet])
    result = planner.dispatch(
        "update_task", {"pet_name": "Mochi", "title": "walk", "duration_minutes": 20, "priority": "nope"}
    )
    assert "error" in result
    assert planner.pets[0].tasks[0].duration_minutes == 30  # unchanged


def test_finish_refused_until_plan_rebuilt():
    planner = tools([Pet("Mochi", "dog")])
    assert "build_plan" in planner.dispatch("finish", {"explanation": "done"})["error"]
    planner.dispatch("build_plan", {})
    assert planner.dispatch("finish", {"explanation": "done"})["ok"]
    planner.dispatch("set_time_budget", {"minutes": 90})
    assert "error" in planner.dispatch("finish", {"explanation": "done"})


def test_working_copy_leaves_input_pets_untouched():
    original = [Pet("Mochi", "dog", [Task("Walk", 30, Priority.HIGH, due_date=DAY)])]
    planner = tools(original)
    planner.dispatch("add_task", dict(pet_name="Mochi", title="Feed", duration_minutes=10,
                                      priority="high", frequency="daily"))
    planner.dispatch("remove_task", {"pet_name": "Mochi", "title": "Walk"})
    assert [t.title for t in original[0].tasks] == ["Walk"]
    assert [t.title for t in planner.pets[0].tasks] == ["Feed"]


# --- The loop ------------------------------------------------------------------


def test_happy_path_builds_pets_and_tasks():
    result, _ = run([
        call("set_time_budget", minutes=120),
        call("add_pet", name="Mochi", animal_type="dog"),
        call("add_task", pet_name="Mochi", title="Morning walk", duration_minutes=40,
             priority="high", frequency="daily"),
        call("add_task", pet_name="Mochi", title="Vet", duration_minutes=60,
             priority="high", frequency="once", fixed_start="10:00"),
        call("build_plan"),
        call("finish", explanation="Walk first, vet at ten."),
    ], minutes=60)

    assert result.stopped_reason == "finished"
    assert result.explanation == "Walk first, vet at ten."
    assert result.available_minutes == 120
    assert [(e.start_time, e.task.title) for e in sorted(result.schedule.entries, key=lambda e: e.start_minutes)] == [
        ("08:00", "Morning walk"), ("10:00", "Vet"),
    ]
    assert result.issues == []
    assert [step.tool for step in result.trace][-2:] == ["build_plan", "finish"]


def test_validation_error_is_fed_back_and_model_can_recover():
    result, llm = run([
        call("add_pet", name="Luna", animal_type="cat"),
        call("add_task", pet_name="Luna", title="Feed", duration_minutes=10,
             priority="critical", frequency="daily"),
        call("add_task", pet_name="Luna", title="Feed", duration_minutes=10,
             priority="high", frequency="daily"),
        call("build_plan"),
        call("finish", explanation="ok"),
    ])
    results = llm.tool_results()
    assert "priority must be" in results[1]["error"]
    assert results[2]["ok"]
    assert result.trace[1].outcome.startswith("error:")
    assert [t.title for t in result.pets[0].tasks] == ["Feed"]


def test_revise_loop_fixes_a_skipped_high_priority_task():
    result, llm = run([
        call("add_pet", name="Mochi", animal_type="dog"),
        call("add_task", pet_name="Mochi", title="Long walk", duration_minutes=50,
             priority="high", frequency="daily"),
        call("add_task", pet_name="Mochi", title="Feed", duration_minutes=10,
             priority="high", frequency="daily"),
        call("add_task", pet_name="Mochi", title="Meds", duration_minutes=5,
             priority="high", frequency="daily"),
        call("build_plan"),  # 30-minute budget: the walk is skipped
        call("update_task", pet_name="Mochi", title="Long walk", duration_minutes=15),
        call("build_plan"),
        call("finish", explanation="Shortened the walk so everything fits."),
    ], minutes=30)

    first_plan = [r for r in llm.tool_results() if "timeline" in r][0]
    assert [t["title"] for t in first_plan["skipped_for_lack_of_time"]] == ["Long walk"]
    assert result.schedule.skipped == []
    assert result.schedule.total_minutes == 30
    assert result.issues == []


def test_post_check_flags_problems_the_model_glossed_over():
    result, _ = run([
        call("add_pet", name="Mochi", animal_type="dog"),
        call("add_task", pet_name="Mochi", title="Walk", duration_minutes=90,
             priority="high", frequency="daily"),
        call("add_task", pet_name="Mochi", title="Vet", duration_minutes=30,
             priority="high", frequency="once", fixed_start="09:00"),
        call("add_task", pet_name="Mochi", title="Groomer", duration_minutes=30,
             priority="high", frequency="once", fixed_start="09:15"),
        call("build_plan"),
        call("finish", explanation="Everything fits perfectly!"),
    ], minutes=80)

    issues = " | ".join(result.issues)
    assert "High-priority 'Walk' for Mochi" in issues
    assert "Conflict: Vet for Mochi" in issues


def test_step_cap_stops_a_runaway_loop():
    result, llm = run([call("add_pet", name="Mochi", animal_type="dog")], max_steps=4)
    assert result.stopped_reason == "max_steps"
    assert len(llm.seen) == 4
    assert any("limit of 4 steps" in issue for issue in result.issues)
    assert [pet.name for pet in result.pets] == ["Mochi"]  # partial work is kept


def test_prose_reply_ends_the_loop_but_still_post_checks():
    result, _ = run([ModelTurn(text="Could you tell me your pets' names?")])
    assert result.stopped_reason == "model_replied"
    assert result.explanation == "Could you tell me your pets' names?"
    assert any("No tasks were created" in issue for issue in result.issues)


def test_opening_message_lists_existing_pets_and_budget():
    pets = [Pet("Luna", "cat", [Task("Feed", 10, Priority.HIGH, due_date=DAY)])]
    _, llm = run([ModelTurn(text="ok")], pets=pets, minutes=45)
    opening = llm.seen[0][0]["text"]
    assert "Luna (cat): Feed (10 min, high)" in opening
    assert "45 minutes" in opening
    assert "plan my day" in opening


def test_parallel_tool_calls_in_one_turn_all_run():
    both = ModelTurn(tool_calls=[
        ToolCall("add_pet", {"name": "Mochi", "animal_type": "dog"}),
        ToolCall("add_pet", {"name": "Luna", "animal_type": "cat"}),
    ])
    result, _ = run([both, ModelTurn(text="done")])
    assert [pet.name for pet in result.pets] == ["Mochi", "Luna"]


# --- Second opinion from the trained task model ----------------------------------


class StubTaskModel:
    """Always says "low", with the given confidence."""

    def __init__(self, confidence: float):
        self.confidence = confidence

    def predict(self, task, species):
        from task_model import TaskPrediction

        return TaskPrediction(Priority.LOW, 10, "daily", {"priority": self.confidence})


SECOND_OPINION_SCRIPT = [
    call("add_pet", name="Mochi", animal_type="dog"),
    call("add_task", pet_name="Mochi", title="Walk", duration_minutes=30, priority="high", frequency="daily"),
    call("add_task", pet_name="Mochi", title="Brush", duration_minutes=10, priority="low", frequency="weekly"),
    call("add_task", pet_name="Mochi", title="Temp", duration_minutes=5, priority="high", frequency="once"),
    call("remove_task", pet_name="Mochi", title="Temp"),
    call("build_plan"),
    call("finish", explanation="ok"),
]


def run_with_model(task_model, pets=None):
    agent = PlanningAgent(FakeLLM(SECOND_OPINION_SCRIPT), task_model=task_model)
    return agent.run("plan", "Jordan", 60, pets, on=DAY)


def test_confident_disagreement_is_reported():
    result = run_with_model(StubTaskModel(0.9))
    # Walk disagrees (high vs low); Brush agrees; Temp was removed again.
    assert [(o.task, o.agent_priority, o.model_priority) for o in result.second_opinions] == [
        ("Walk", "high", "low")
    ]
    assert result.second_opinions[0].confidence == 0.9


def test_unsure_model_stays_quiet():
    assert run_with_model(StubTaskModel(0.5)).second_opinions == []


def test_owner_entered_tasks_are_not_second_guessed():
    pets = [Pet("Rex", "dog", [Task("Vet", 60, Priority.HIGH, due_date=DAY)])]
    result = run_with_model(StubTaskModel(0.99), pets=pets)
    assert all(o.pet != "Rex" for o in result.second_opinions)


def test_no_model_means_no_second_opinions():
    assert run_with_model(None).second_opinions == []


def test_real_trained_model_agrees_with_sensible_priorities():
    from task_model import TaskModel

    script = [
        call("add_pet", name="Luna", animal_type="cat"),
        call("add_task", pet_name="Luna", title="Kidney meds", duration_minutes=5,
             priority="high", frequency="daily"),
        call("add_task", pet_name="Luna", title="Laser pointer play", duration_minutes=10,
             priority="high", frequency="daily"),  # the model should query this one
        call("build_plan"),
        call("finish", explanation="ok"),
    ]
    agent = PlanningAgent(FakeLLM(script), task_model=TaskModel.train())
    result = agent.run("plan", "Jordan", 60, None, on=DAY)
    flagged = [o.task for o in result.second_opinions]
    assert "Kidney meds" not in flagged
    assert flagged == ["Laser pointer play"]


# --- Progress reporting and Gemini settings ----------------------------------


def test_progress_reports_each_step_and_tool_call():
    seen: list[tuple[int, str]] = []
    agent = PlanningAgent(FakeLLM([
        call("add_pet", name="Mochi", animal_type="dog"),
        call("add_task", pet_name="Mochi", title="Walk", duration_minutes=30,
             priority="urgent", frequency="daily"),
        call("build_plan"),
        call("finish", explanation="Done."),
    ]))
    agent.run("plan my day", "Jordan", 60, on=DAY, on_progress=lambda s, m: seen.append((s, m)))

    assert seen[0] == (1, "Waiting for Gemini...")
    assert (1, "Added Mochi") in seen
    rejected = [m for s, m in seen if s == 2 and m.startswith("Added 'Walk' for Mochi (rejected:")]
    assert rejected, seen
    assert seen[-1] == (4, "Writing up the plan")


@pytest.mark.parametrize(
    "setting, expected",
    [(None, "LOW"), ("minimal", "MINIMAL"), ("HIGH", "HIGH"), ("default", None)],
)
def test_gemini_client_speed_settings(monkeypatch, setting, expected):
    pytest.importorskip("google.genai")
    from agent import GEMINI_RETRY_ATTEMPTS, GEMINI_TIMEOUT_MS, GeminiClient

    if setting is None:
        monkeypatch.delenv("PAWPAL_GEMINI_THINKING", raising=False)
    else:
        monkeypatch.setenv("PAWPAL_GEMINI_THINKING", setting)
    client = GeminiClient(api_key="test-key")  # builds the client; no network call

    level = client._thinking.thinking_level if client._thinking else None
    assert (getattr(level, "value", level) if level else None) == expected
    options = client._client._api_client._http_options
    assert options.timeout == GEMINI_TIMEOUT_MS
    assert options.retry_options.attempts == GEMINI_RETRY_ATTEMPTS


class Overloaded(Exception):
    """Shaped like google.genai.errors.APIError: carries an HTTP status code."""

    def __init__(self, code: int):
        super().__init__(f"{code} UNAVAILABLE")
        self.code = code


class FakeModels:
    """Stands in for client.models: fails for the models listed in `down`."""

    def __init__(self, down: dict[str, int]):
        self.down = down
        self.tried: list[str] = []

    def generate_content(self, model, contents, config):
        from google.genai import types

        self.tried.append(model)
        if model in self.down:
            raise Overloaded(self.down[model])
        part = types.Part.from_text(text=f"answer from {model}")
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
        )


FIRST_TURN = [{"role": "user", "text": "plan my day"}]


@pytest.fixture
def gemini(monkeypatch):
    """A GeminiClient whose SDK calls go to FakeModels, with a clean cooldown."""
    pytest.importorskip("google.genai")
    import agent

    monkeypatch.setattr(agent, "_overloaded_until", {})
    monkeypatch.delenv("PAWPAL_GEMINI_MODEL", raising=False)
    monkeypatch.setenv("PAWPAL_GEMINI_FALLBACK_MODELS", "lite-a,lite-b")

    def make(down: dict[str, int]):
        client = agent.GeminiClient(api_key="test-key")
        client._client = SimpleNamespace(models=FakeModels(down))
        return client

    return make


def test_gemini_falls_back_when_the_main_model_is_overloaded(gemini):
    from agent import DEFAULT_GEMINI_MODEL

    client = gemini({DEFAULT_GEMINI_MODEL: 503, "lite-a": 504})
    assert client.next_step("system", FIRST_TURN, []).text == "answer from lite-b"
    assert client._client.models.tried == [DEFAULT_GEMINI_MODEL, "lite-a", "lite-b"]
    # A new client (the next request) skips the models that are cooling down.
    assert gemini({}).model == "lite-b"


def test_gemini_does_not_switch_models_mid_run(gemini):
    from agent import DEFAULT_GEMINI_MODEL, LLMError

    client = gemini({DEFAULT_GEMINI_MODEL: 503})
    history = [
        {"role": "user", "text": "plan my day"},
        {"role": "model", "turn": ModelTurn(text="thinking")},
    ]
    with pytest.raises(LLMError, match="503"):
        client.next_step("system", history, [])
    assert client._client.models.tried == [DEFAULT_GEMINI_MODEL]


def test_gemini_reports_errors_that_are_not_overload(gemini):
    from agent import DEFAULT_GEMINI_MODEL, LLMError

    client = gemini({DEFAULT_GEMINI_MODEL: 400})
    with pytest.raises(LLMError, match="400"):
        client.next_step("system", FIRST_TURN, [])
    assert client._client.models.tried == [DEFAULT_GEMINI_MODEL]
