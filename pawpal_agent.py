"""PawPal+ planning agent: natural language in, a checked daily plan out.

The owner describes their pets and their day in plain English. The agent then
works in a loop of plan, act and check:

    search the care guide (RAG)  ->  add pets and tasks  ->  build_plan
            ^                                                   |
            +---- revise priorities / durations if it fails ----+

The deterministic Scheduler in pawpal_system stays the source of truth. The
model never writes a schedule itself: it can only call the tools below, every
tool validates its input, and when the model says it is done the plan is
rebuilt once more and checked independently (see PlanningAgent._post_check).

The model sits behind a one-method LLMClient protocol, so the loop can be driven
by Gemini in the app and by a scripted fake in the tests.
"""

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Protocol

from pawpal_guardrails import (
    MAX_EXPLANATION_CHARS,
    MAX_REQUEST_CHARS,
    clean_model_text,
    require_input,
)
from pawpal_logging import get_logger, preview
from pawpal_rag import KnowledgeBase, Passage
from pawpal_system import (
    FREQUENCY_STEPS,
    Owner,
    Pet,
    Priority,
    Schedule,
    Scheduler,
    Task,
    format_time,
    parse_time,
)

# Limits match the manual form in app.py, so the agent cannot create anything a
# person could not have typed in by hand.
SPECIES = ("dog", "cat", "other")
FREQUENCIES = ("once", *FREQUENCY_STEPS)
MIN_DURATION, MAX_DURATION = 1, 240
MIN_BUDGET, MAX_BUDGET = 5, 600
MAX_TASKS = 30
MAX_STEPS = 15
# Guardrails on the loop itself, on top of MAX_STEPS:
MAX_CALLS_PER_TURN = 12  # tool calls beyond this in one turn are refused
MAX_REPEATED_ERRORS = 3  # the same failing call this many times ends the run

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Speed settings for every Gemini call. Each plan is many round trips, so
# per-call latency adds up:
#   * thinking: picking the next tool call needs little reasoning, and the
#     model's default level made multi-step plans take minutes. Override with
#     PAWPAL_GEMINI_THINKING (minimal, low, medium, high, or "default" to leave
#     it to the model).
#   * timeout + retries: a stuck call fails in under a minute instead of
#     hanging, and a "high demand" 503 or a 429 is retried twice, quickly.
#   * fallback: when the main model is overloaded (429/5xx or a timeout), switch
#     to a lighter model rather than fail. Lite models were answering in under
#     a second while every full Flash model returned 503s. Override with
#     PAWPAL_GEMINI_FALLBACK_MODELS (comma-separated; empty turns it off). An
#     overloaded model is skipped for OVERLOAD_COOLDOWN_SECONDS so each new
#     request doesn't wait on it again.
DEFAULT_GEMINI_THINKING = "low"
DEFAULT_GEMINI_FALLBACKS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
GEMINI_TIMEOUT_MS = 30_000
GEMINI_RETRY_ATTEMPTS = 2  # including the first try; the fallback is the next resort
GEMINI_RETRY_CODES = (429, 500, 503)
OVERLOAD_CODES = (429, 500, 502, 503, 504)
OVERLOAD_COOLDOWN_SECONDS = 300
_overloaded_until: dict[str, float] = {}  # model -> monotonic time it may be tried again

log = get_logger("agent")


# --- Provider-neutral conversation types ------------------------------------


@dataclass
class ToolCall:
    """One function call the model asked for."""

    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ModelTurn:
    """What the model returned on one step: any text, any tool calls, and the
    provider's raw message so it can be replayed verbatim (Gemini requires its
    thought signatures to come back unchanged)."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMClient(Protocol):
    """Anything that can take the conversation so far and produce the next turn.

    messages is a list of dicts in one of three shapes:
        {"role": "user", "text": str}
        {"role": "model", "turn": ModelTurn}
        {"role": "tool", "results": [(ToolCall, dict), ...]}
    """

    def next_step(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> ModelTurn: ...


class LLMError(RuntimeError):
    """The model could not produce a usable turn (blocked, empty, API failure)."""


# --- Tools -----------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "name": "search_care_guide",
        "description": (
            "Search the PawPal+ pet care guide. Use it before choosing a task's "
            "duration, priority or frequency. Returns the best matching passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "e.g. 'husky daily exercise'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "set_time_budget",
        "description": (
            "Set how many minutes the owner has for pet care today. Only call this "
            "if the owner states their available time."
        ),
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer"}},
            "required": ["minutes"],
        },
    },
    {
        "name": "add_pet",
        "description": "Add a pet. animal_type must be 'dog', 'cat' or 'other'.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "animal_type": {"type": "string", "enum": list(SPECIES)},
            },
            "required": ["name", "animal_type"],
        },
    },
    {
        "name": "add_task",
        "description": (
            "Add a care task for an existing pet. Use fixed_start (HH:MM) only for "
            "things that must happen at a set time, such as a vet appointment or a "
            "timed medication dose; leave it out otherwise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string"},
                "title": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "frequency": {"type": "string", "enum": list(FREQUENCIES)},
                "fixed_start": {"type": "string", "description": "HH:MM, optional"},
            },
            "required": ["pet_name", "title", "duration_minutes", "priority", "frequency"],
        },
    },
    {
        "name": "update_task",
        "description": (
            "Change an outstanding task's duration, priority, frequency or fixed "
            "start. Pass only the fields to change; pass fixed_start as '' to make "
            "it flexible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string"},
                "title": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "frequency": {"type": "string", "enum": list(FREQUENCIES)},
                "fixed_start": {"type": "string"},
            },
            "required": ["pet_name", "title"],
        },
    },
    {
        "name": "remove_task",
        "description": "Remove an outstanding task from a pet.",
        "parameters": {
            "type": "object",
            "properties": {"pet_name": {"type": "string"}, "title": {"type": "string"}},
            "required": ["pet_name", "title"],
        },
    },
    {
        "name": "build_plan",
        "description": (
            "Run the PawPal+ scheduler on the current pets and tasks. Returns the "
            "timeline, the tasks skipped for lack of time, and any clashes between "
            "fixed-time tasks."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": (
            "Finish once build_plan has been run on the final tasks. explanation "
            "is shown to the owner: say why the plan looks the way it does and "
            "what, if anything, did not fit."
        ),
        "parameters": {
            "type": "object",
            "properties": {"explanation": {"type": "string"}},
            "required": ["explanation"],
        },
    },
]


class ToolError(ValueError):
    """Bad tool input. The message goes back to the model so it can correct itself."""


class PlannerTools:
    """The only way the agent can touch PawPal+ data.

    Works on a deep copy of the pets it is given, so nothing the agent does is
    visible to the app until the caller chooses to commit PlannerTools.pets.

    Every tool returns a JSON-safe dict. Bad input never raises out of
    dispatch(): it comes back as {"error": "..."} for the model to read and fix.
    """

    def __init__(
        self,
        knowledge: KnowledgeBase,
        owner_name: str,
        available_minutes: int,
        pets: list[Pet] | None = None,
        on: date | None = None,
    ):
        self.knowledge = knowledge
        self.owner_name = owner_name
        self.available_minutes = available_minutes
        self.pets: list[Pet] = copy.deepcopy(pets or [])
        self.on = on or date.today()
        self.scheduler = Scheduler()
        self.citations: list[Passage] = []
        self.last_schedule: Schedule | None = None
        # Every task the agent added, for the trained model's second opinion.
        self.created: list[tuple[Pet, Task]] = []
        # Set by every change and cleared by build_plan, so finish() can insist
        # the model has actually looked at the plan it is about to hand over.
        self.plan_is_stale = True

    # Dispatch ---------------------------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool by name, turning any validation failure into an error dict."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return handler(**(args or {}))
        except ToolError as exc:
            return {"error": str(exc)}
        except TypeError as exc:  # missing or unexpected arguments
            return {"error": f"Bad arguments for {name}: {exc}"}

    # Validation helpers -------------------------------------------------------

    def owner(self) -> Owner:
        """An Owner over the working pets, for the scheduler."""
        return Owner(self.owner_name, pets=self.pets, available_minutes=self.available_minutes)

    def _pet(self, name: str) -> Pet:
        wanted = str(name).strip().lower()
        for pet in self.pets:
            if pet.name.lower() == wanted:
                return pet
        known = ", ".join(pet.name for pet in self.pets) or "none yet"
        raise ToolError(f"No pet named '{name}'. Known pets: {known}. Add it with add_pet first.")

    def _task(self, pet: Pet, title: str) -> Task:
        wanted = str(title).strip().lower()
        for task in pet.tasks:
            if not task.is_completed and task.title.lower() == wanted:
                return task
        known = ", ".join(t.title for t in pet.tasks if not t.is_completed) or "none"
        raise ToolError(f"{pet.name} has no outstanding task '{title}'. Their tasks: {known}.")

    @staticmethod
    def _int(value: Any, label: str, low: int, high: int) -> int:
        # Models sometimes send 20.0 or "20"; accept those, reject 20.5 and "twenty".
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ToolError(f"{label} must be a whole number, got {value!r}.") from None
        if not number.is_integer():
            raise ToolError(f"{label} must be a whole number, got {value!r}.")
        if not low <= number <= high:
            raise ToolError(f"{label} must be between {low} and {high}, got {int(number)}.")
        return int(number)

    @staticmethod
    def _priority(value: Any) -> Priority:
        try:
            return Priority[str(value).strip().upper()]
        except KeyError:
            raise ToolError(f"priority must be low, medium or high, got {value!r}.") from None

    @staticmethod
    def _frequency(value: Any) -> str:
        text = str(value).strip().lower()
        if text not in FREQUENCIES:
            raise ToolError(f"frequency must be one of {', '.join(FREQUENCIES)}, got {value!r}.")
        return text

    @staticmethod
    def _fixed_start(value: Any) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        minutes = parse_time(value)
        if minutes is None:
            raise ToolError(f"fixed_start must be HH:MM in 24-hour time, got {value!r}.")
        return minutes

    # Tools ---------------------------------------------------------------------

    def tool_search_care_guide(self, query: str) -> dict:
        passages = self.knowledge.search(str(query), k=3)
        for passage in passages:
            if all(p.cite() != passage.cite() for p in self.citations):
                self.citations.append(passage)
        if not passages:
            return {"results": [], "note": "Nothing in the guide matches; try other words."}
        return {
            "results": [
                {"source": p.cite(), "text": p.text, "score": p.score} for p in passages
            ]
        }

    def tool_set_time_budget(self, minutes: Any) -> dict:
        self.available_minutes = self._int(minutes, "minutes", MIN_BUDGET, MAX_BUDGET)
        self.plan_is_stale = True
        return {"ok": True, "available_minutes": self.available_minutes}

    def tool_add_pet(self, name: str, animal_type: str) -> dict:
        name = str(name).strip()
        species = str(animal_type).strip().lower()
        if not name:
            raise ToolError("Pet name cannot be empty.")
        if species not in SPECIES:
            raise ToolError(f"animal_type must be one of {', '.join(SPECIES)}, got {animal_type!r}.")
        if any(pet.name.lower() == name.lower() for pet in self.pets):
            raise ToolError(f"{name} already exists; add tasks to it instead.")
        self.pets.append(Pet(name, species))
        self.plan_is_stale = True
        return {"ok": True, "pet": name, "animal_type": species}

    def tool_add_task(
        self,
        pet_name: str,
        title: str,
        duration_minutes: Any,
        priority: Any,
        frequency: Any = "once",
        fixed_start: Any = None,
    ) -> dict:
        pet = self._pet(pet_name)
        title = str(title).strip()
        if not title:
            raise ToolError("Task title cannot be empty.")
        if any(t.title.lower() == title.lower() and not t.is_completed for t in pet.tasks):
            raise ToolError(f"{pet.name} already has '{title}'; use update_task to change it.")
        if sum(len(p.tasks) for p in self.pets) >= MAX_TASKS:
            raise ToolError(f"Task limit of {MAX_TASKS} reached; remove or merge tasks instead.")
        task = Task(
            title,
            self._int(duration_minutes, "duration_minutes", MIN_DURATION, MAX_DURATION),
            self._priority(priority),
            frequency=self._frequency(frequency),
            due_date=self.on,
            required_start=self._fixed_start(fixed_start),
        )
        pet.add_task(task)
        self.created.append((pet, task))
        self.plan_is_stale = True
        return {"ok": True, "added": _task_view(pet, task)}

    def tool_update_task(
        self,
        pet_name: str,
        title: str,
        duration_minutes: Any = None,
        priority: Any = None,
        frequency: Any = None,
        fixed_start: Any = None,
    ) -> dict:
        pet = self._pet(pet_name)
        task = self._task(pet, title)
        # Validate everything before changing anything, so a half-bad call
        # leaves the task as it was.
        changes: dict[str, Any] = {}
        if duration_minutes is not None:
            changes["duration_minutes"] = self._int(
                duration_minutes, "duration_minutes", MIN_DURATION, MAX_DURATION
            )
        if priority is not None:
            changes["priority"] = self._priority(priority)
        if frequency is not None:
            changes["frequency"] = self._frequency(frequency)
        if fixed_start is not None:
            changes["required_start"] = self._fixed_start(fixed_start)
        if not changes:
            raise ToolError("Nothing to update: pass at least one field to change.")
        for attribute, value in changes.items():
            setattr(task, attribute, value)
        self.plan_is_stale = True
        return {"ok": True, "updated": _task_view(pet, task)}

    def tool_remove_task(self, pet_name: str, title: str) -> dict:
        pet = self._pet(pet_name)
        task = self._task(pet, title)
        pet.tasks = [item for item in pet.tasks if item is not task]
        self.plan_is_stale = True
        return {"ok": True, "removed": task.title, "pet": pet.name}

    def tool_build_plan(self) -> dict:
        self.last_schedule = self.scheduler.build_plan(self.owner(), on=self.on)
        self.plan_is_stale = False
        return summarize_schedule(self.last_schedule, self.available_minutes)

    def tool_finish(self, explanation: str) -> dict:
        if self.plan_is_stale:
            raise ToolError("Tasks changed since the last build_plan. Call build_plan, check it, then finish.")
        return {"ok": True, "explanation": str(explanation)}


def _task_view(pet: Pet, task: Task) -> dict:
    """A compact JSON-safe description of one task, for tool results."""
    return {
        "pet": pet.name,
        "title": task.title,
        "minutes": task.duration_minutes,
        "priority": task.priority.name.lower(),
        "frequency": task.frequency,
        "fixed_start": format_time(task.required_start) if task.is_pinned else None,
    }


def summarize_schedule(schedule: Schedule, available_minutes: int) -> dict:
    """The parts of a Schedule the model needs to judge it, as plain JSON."""
    return {
        "available_minutes": available_minutes,
        "planned_minutes": schedule.total_minutes,
        "timeline": [
            {
                "time": f"{entry.start_time}-{entry.end_time}",
                **_task_view(entry.pet, entry.task),
            }
            for entry in sorted(schedule.entries, key=lambda e: e.start_minutes)
        ],
        "skipped_for_lack_of_time": [_task_view(pet, task) for pet, task in schedule.skipped],
        "conflicts": list(schedule.warnings),
    }


# --- The agent ---------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the planning assistant inside PawPal+, a pet care planner. The owner
describes their pets and their day; you turn that into pets and care tasks using
the tools, then check the result with the scheduler.

How to work:
1. Use search_care_guide before deciding any task's duration, priority or
   frequency. Base your numbers on what the guide says, adjusted for what the
   owner told you. Do not invent medical advice; for medication, use the times
   the owner gives.
2. If the owner states how much time they have, call set_time_budget.
3. Add any pets that do not exist yet, then add their tasks. Use fixed_start
   only for things that must happen at a set time (appointments, timed doses).
   Use frequency "daily" for routine care, "once" for one-off events.
4. Call build_plan. If a high-priority task was skipped or there are conflicts,
   revise: shorten or lower the priority of less important tasks, or move a
   fixed time if the owner allowed it. Then call build_plan again.
5. When the plan is the best you can do, call finish with a short explanation
   for the owner (a few sentences): what is in the plan, why, and anything that
   did not fit. Mention which guide sections you relied on.

The scheduler places fixed-time tasks first, then everything else highest
priority first and shortest first on ties, skipping whatever exceeds the time
budget. Keep task titles short, like "Morning walk" or "Evening kidney meds".
"""


@dataclass
class TraceStep:
    """One tool call the agent made, for the "Agent steps" view."""

    step: int
    tool: str
    args: dict[str, Any]
    outcome: str  # short human-readable result


class TaskLabeller(Protocol):
    """A trained model that can predict a task's priority (see pawpal_model)."""

    def predict(self, task: str, species: str) -> Any: ...


@dataclass
class SecondOpinion:
    """A task where the trained model confidently disagrees with the agent."""

    pet: str
    task: str
    agent_priority: str
    model_priority: str
    confidence: float


# The trained model must be at least this sure before it contradicts the agent;
# below it, disagreements are mostly the model's own noise.
SECOND_OPINION_CONFIDENCE = 0.7


@dataclass
class AgentResult:
    """Everything a run produced. pets is the working copy, ready to commit."""

    pets: list[Pet]
    available_minutes: int
    schedule: Schedule
    explanation: str
    citations: list[Passage]
    trace: list[TraceStep]
    issues: list[str]  # everything the post-check found
    stopped_reason: str  # "finished", "model_replied", "max_steps" or "stuck"
    # The subset of issues about how the run went (step cap, no tasks, guide not
    # used) rather than about the plan itself. The app shows the plan's own
    # problems live, so it only needs these from the agent.
    process_issues: list[str] = field(default_factory=list)
    # Tasks whose priority the trained task model would have set differently.
    # Advisory only: the agent's choice stands.
    second_opinions: list[SecondOpinion] = field(default_factory=list)
    run_id: str = ""  # ties this run to its lines in logs/pawpal.log


class PlanningAgent:
    """Drives an LLMClient through the plan-act-check loop over PlannerTools."""

    def __init__(
        self,
        llm: LLMClient,
        knowledge: KnowledgeBase,
        max_steps: int = MAX_STEPS,
        task_model: TaskLabeller | None = None,
    ):
        self.llm = llm
        self.knowledge = knowledge
        self.max_steps = max_steps
        self.task_model = task_model

    def run(
        self,
        request: str,
        owner_name: str,
        available_minutes: int,
        existing_pets: list[Pet] | None = None,
        on: date | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> AgentResult:
        """Turn the owner's description into pets, tasks and a checked plan.

        on_progress, if given, is called as on_progress(step, message) before
        each model call and after each tool call, so a UI can show the run
        moving instead of a bare spinner.

        The pets passed in are never modified; the result carries a new list.
        LLMError propagates, so a failed call leaves the caller's data alone.
        GuardrailError is raised, before any model call, if the request is
        empty, too long or reads as prompt injection.
        """
        run_id = uuid.uuid4().hex[:8]
        request = require_input(request, MAX_REQUEST_CHARS, "request")
        tools = PlannerTools(self.knowledge, owner_name, available_minutes, existing_pets, on)
        log.info(
            "[run %s] start: %d existing pet(s), %d min, request %r",
            run_id, len(tools.pets), available_minutes, preview(request),
        )
        messages: list[dict] = [{"role": "user", "text": self._opening(request, tools)}]
        trace: list[TraceStep] = []
        explanation = ""
        stopped_reason = "max_steps"
        error_counts: dict[str, int] = {}

        def progress(step: int, message: str) -> None:
            if on_progress is not None:
                on_progress(step, message)

        for step in range(1, self.max_steps + 1):
            progress(step, "Waiting for Gemini...")
            started = time.perf_counter()
            try:
                turn = self.llm.next_step(SYSTEM_PROMPT, messages, TOOL_SPECS)
            except LLMError as exc:
                log.error("[run %s] step %d: model call failed: %s", run_id, step, exc)
                raise
            log.info(
                "[run %s] step %d: model replied in %.2fs with %d tool call(s)",
                run_id, step, time.perf_counter() - started, len(turn.tool_calls),
            )
            messages.append({"role": "model", "turn": turn})

            if not turn.tool_calls:
                # The model answered in prose instead of calling finish. Accept
                # it as the explanation; the post-check still runs.
                explanation = turn.text.strip()
                stopped_reason = "model_replied"
                break

            results = []
            finished = False
            for index, call in enumerate(turn.tool_calls):
                if index >= MAX_CALLS_PER_TURN:
                    result = {"error": f"At most {MAX_CALLS_PER_TURN} tool calls per turn; this one was not run."}
                else:
                    result = tools.dispatch(call.name, call.args)
                results.append((call, result))
                trace.append(TraceStep(step, call.name, dict(call.args or {}), _outcome(result)))
                progress(step, describe_call(call, result))
                if "error" in result:
                    log.warning("[run %s] step %d: %s %s -> %s", run_id, step, call.name,
                                json.dumps(call.args, default=str), result["error"])
                    signature = f"{call.name}:{json.dumps(call.args, sort_keys=True, default=str)}"
                    error_counts[signature] = error_counts.get(signature, 0) + 1
                else:
                    log.info("[run %s] step %d: %s -> %s", run_id, step, call.name, _outcome(result))
                if call.name == "finish" and "error" not in result:
                    explanation = result["explanation"]
                    finished = True
            messages.append({"role": "tool", "results": results})
            if finished:
                stopped_reason = "finished"
                break
            if max(error_counts.values(), default=0) >= MAX_REPEATED_ERRORS:
                # The model keeps making the same failing call; more steps
                # would only repeat it and spend more of the budget.
                log.warning("[run %s] stopped: same failing call %d times", run_id, MAX_REPEATED_ERRORS)
                stopped_reason = "stuck"
                break

        explanation = clean_model_text(explanation, MAX_EXPLANATION_CHARS)
        schedule, process_issues, plan_issues = self._post_check(tools, stopped_reason)
        second_opinions = self._second_opinions(tools)
        log.info(
            "[run %s] done (%s): %d pet(s), %d task(s), %d citation(s), %d planned, %d skipped",
            run_id, stopped_reason, len(tools.pets), sum(len(p.tasks) for p in tools.pets),
            len(tools.citations), len(schedule.entries), len(schedule.skipped),
        )
        for issue in process_issues + plan_issues:
            log.warning("[run %s] post-check: %s", run_id, issue)
        for opinion in second_opinions:
            log.info(
                "[run %s] second opinion: %s/%s agent=%s model=%s (%.0f%%)", run_id, opinion.pet,
                opinion.task, opinion.agent_priority, opinion.model_priority, opinion.confidence * 100,
            )
        return AgentResult(
            pets=tools.pets,
            available_minutes=tools.available_minutes,
            schedule=schedule,
            explanation=explanation,
            citations=tools.citations,
            trace=trace,
            issues=process_issues + plan_issues,
            stopped_reason=stopped_reason,
            process_issues=process_issues,
            second_opinions=second_opinions,
            run_id=run_id,
        )

    def _second_opinions(self, tools: PlannerTools) -> list[SecondOpinion]:
        """Where the trained task model confidently disagrees on priority.

        Only tasks the agent added and kept are checked; tasks the owner
        entered by hand are theirs to judge.
        """
        if self.task_model is None:
            return []
        opinions = []
        for pet, task in tools.created:
            if not any(item is task for item in pet.tasks):
                continue  # added, then removed again
            prediction = self.task_model.predict(task.title, pet.animal_type)
            confidence = prediction.confidence["priority"]
            if prediction.priority != task.priority and confidence >= SECOND_OPINION_CONFIDENCE:
                opinions.append(
                    SecondOpinion(
                        pet.name,
                        task.title,
                        task.priority.name.lower(),
                        prediction.priority.name.lower(),
                        confidence,
                    )
                )
        return opinions

    @staticmethod
    def _opening(request: str, tools: PlannerTools) -> str:
        """The first user message: the request plus what already exists."""
        if tools.pets:
            lines = []
            for pet in tools.pets:
                outstanding = [t for t in pet.tasks if not t.is_completed]
                listed = "; ".join(
                    f"{t.title} ({t.duration_minutes} min, {t.priority.name.lower()})"
                    for t in outstanding
                ) or "no tasks"
                lines.append(f"- {pet.name} ({pet.animal_type}): {listed}")
            existing = "Pets already in PawPal+:\n" + "\n".join(lines)
        else:
            existing = "No pets in PawPal+ yet."
        return (
            f"Owner: {tools.owner_name}. Time currently set aside today: "
            f"{tools.available_minutes} minutes, starting 08:00. Today is {tools.on}.\n"
            f"{existing}\n\nOwner's request:\n{request.strip()}"
        )

    def _post_check(
        self, tools: PlannerTools, stopped_reason: str
    ) -> tuple[Schedule, list[str], list[str]]:
        """Rebuild the plan independently of the model and list what is wrong with it.

        This is the reliability layer: the model's own "looks good" is never the
        last word. Whatever it claimed, the final pets go through the real
        scheduler once more and the issues are derived from that result.

        Returns:
            The rebuilt schedule, issues with the run itself, and issues with
            the plan (skipped high-priority tasks and fixed-time clashes).
        """
        schedule = tools.scheduler.build_plan(tools.owner(), on=tools.on)
        process: list[str] = []
        plan: list[str] = []
        if stopped_reason == "max_steps":
            process.append(
                f"The assistant hit its limit of {self.max_steps} steps before finishing;"
                " the tasks below are what it had so far."
            )
        if stopped_reason == "stuck":
            process.append(
                "The assistant kept repeating a step that failed, so it was stopped;"
                " the tasks below are what it had so far."
            )
        if not any(pet.tasks for pet in tools.pets):
            process.append("No tasks were created. Try describing your pets' routine in more detail.")
        for pet, task in schedule.skipped:
            if task.priority is Priority.HIGH:
                plan.append(
                    f"High-priority '{task.title}' for {pet.name} ({task.duration_minutes} min)"
                    " did not fit in the time available."
                )
        plan.extend(f"Conflict: {warning}" for warning in schedule.warnings)
        if any(pet.tasks for pet in tools.pets) and not tools.citations:
            process.append(
                "The assistant did not consult the care guide, so durations and"
                " priorities are its own estimates. Review them before relying on the plan."
            )
        return schedule, process, plan


def _outcome(result: dict) -> str:
    """A one-line summary of a tool result for the trace."""
    if "error" in result:
        return f"error: {result['error']}"
    if "results" in result:
        cites = [r["source"] for r in result["results"]]
        return "found " + (", ".join(cites) if cites else "nothing")
    if "timeline" in result:
        return (
            f"{len(result['timeline'])} planned ({result['planned_minutes']}/"
            f"{result['available_minutes']} min), {len(result['skipped_for_lack_of_time'])}"
            f" skipped, {len(result['conflicts'])} conflict(s)"
        )
    shown = {k: v for k, v in result.items() if k != "ok"}
    return "ok " + json.dumps(shown, default=str) if shown else "ok"


def describe_call(call: ToolCall, result: dict) -> str:
    """A short, owner-friendly line for one tool call, for live progress."""
    args = call.args or {}
    title, pet = args.get("title", ""), args.get("pet_name", "")
    text = {
        "search_care_guide": f"Searched the care guide for '{args.get('query', '')}'",
        "set_time_budget": f"Set the time budget to {args.get('minutes')} min",
        "add_pet": f"Added {args.get('name', 'a pet')}",
        "add_task": f"Added '{title}' for {pet}",
        "update_task": f"Updated '{title}' for {pet}",
        "remove_task": f"Removed '{title}' for {pet}",
        "build_plan": "Built the schedule",
        "finish": "Writing up the plan",
    }.get(call.name, call.name)
    return f"{text} (rejected: {result['error']})" if "error" in result else text


# --- Gemini adapter --------------------------------------------------------------


def gemini_api_key() -> str | None:
    """The Gemini key from the environment, under either name the SDK accepts."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiClient:
    """LLMClient backed by the Gemini API via the google-genai SDK.

    Automatic function calling is disabled: the SDK would otherwise run the
    tools itself, bypassing the agent's step cap, trace and guardrails.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # imported lazily so tests need no SDK or key
        from google.genai import types

        key = api_key or gemini_api_key()
        if not key:
            raise LLMError("Set GEMINI_API_KEY to use the AI planner.")
        self._types = types
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=GEMINI_RETRY_ATTEMPTS,
                    initial_delay=1.0,
                    max_delay=4.0,
                    http_status_codes=list(GEMINI_RETRY_CODES),
                ),
            ),
        )
        primary = model or os.environ.get("PAWPAL_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        fallbacks = os.environ.get("PAWPAL_GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)
        self.models = list(dict.fromkeys(
            [primary] + [name.strip() for name in fallbacks.split(",") if name.strip()]
        ))
        # Start with the first model not cooling down after an overload.
        now = time.monotonic()
        self.model = next(
            (name for name in self.models if _overloaded_until.get(name, 0) <= now), self.models[-1]
        )
        level = os.environ.get("PAWPAL_GEMINI_THINKING", DEFAULT_GEMINI_THINKING).strip().upper()
        self._thinking = (
            None if level in ("", "DEFAULT") else types.ThinkingConfig(thinking_level=level)
        )

    def _generate(self, can_switch: bool, **request):
        """generate_content on the current model, moving down the fallback list
        when a model is overloaded. Sticks with the model that worked, so one
        planning run uses one model (Gemini's thought signatures from one model
        are not replayed to another): can_switch is False once a run has started.
        """
        while True:
            try:
                return self._client.models.generate_content(model=self.model, **request)
            except Exception as exc:  # network, auth, quota, bad model name
                code = getattr(exc, "code", None)
                timed_out = "timeout" in type(exc).__name__.lower()
                position = self.models.index(self.model)
                if (code in OVERLOAD_CODES or timed_out) and can_switch and position + 1 < len(self.models):
                    _overloaded_until[self.model] = time.monotonic() + OVERLOAD_COOLDOWN_SECONDS
                    next_model = self.models[position + 1]
                    log.warning("%s unavailable (%s); falling back to %s", self.model,
                                code or type(exc).__name__, next_model)
                    self.model = next_model
                    continue
                raise LLMError(f"Gemini request failed ({self.model}): {exc}") from exc

    def next_step(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=self._thinking,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=spec["name"],
                            description=spec["description"],
                            parameters_json_schema=spec["parameters"],
                        )
                        for spec in tools
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._generate(
            can_switch=not any(message["role"] == "model" for message in messages),
            contents=[self._content(message) for message in messages],
            config=config,
        )

        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        if content is None or not content.parts:
            reason = getattr(candidate, "finish_reason", None) if candidate else "no candidates"
            raise LLMError(f"Gemini returned an empty response ({reason}).")

        text = "".join(
            part.text for part in content.parts if part.text and not getattr(part, "thought", False)
        )
        calls = [
            ToolCall(part.function_call.name, dict(part.function_call.args or {}), part.function_call.id)
            for part in content.parts
            if part.function_call
        ]
        return ModelTurn(text=text, tool_calls=calls, raw=content)

    def generate(self, system: str, prompt: str) -> str:
        """Plain text generation with no tools, for Ask the care guide."""
        types = self._types
        response = self._generate(
            can_switch=True,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, thinking_config=self._thinking
            ),
        )
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else None
        text = "".join(
            part.text for part in (parts or []) if part.text and not getattr(part, "thought", False)
        )
        if not text.strip():
            reason = getattr(candidate, "finish_reason", None) if candidate else "no candidates"
            raise LLMError(f"Gemini returned an empty response ({reason}).")
        return text

    def _content(self, message: dict):
        """Translate one provider-neutral message into a Gemini Content."""
        types = self._types
        role = message["role"]
        if role == "user":
            return types.Content(role="user", parts=[types.Part.from_text(text=message["text"])])
        if role == "model":
            turn: ModelTurn = message["turn"]
            if turn.raw is not None:
                return turn.raw  # replay verbatim, keeping thought signatures
            parts = [types.Part.from_text(text=turn.text)] if turn.text else []
            parts += [
                types.Part(function_call=types.FunctionCall(name=c.name, args=c.args, id=c.id))
                for c in turn.tool_calls
            ]
            return types.Content(role="model", parts=parts)
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=result
                    )
                )
                for call, result in message["results"]
            ],
        )
