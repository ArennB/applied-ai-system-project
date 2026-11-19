"""End-to-end tests of the Streamlit app, driven headlessly with AppTest.

No API key or network is needed: the AI test swaps GeminiClient for the
scripted FakeLLM from test_agent.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent
from test_agent import FakeLLM, call

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return AppTest.from_file(APP, default_timeout=30).run()


def labelled(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def add_pet(at: AppTest, name: str) -> None:
    labelled(at.sidebar.text_input, "Pet name").input(name)
    labelled(at.sidebar.button, "Add pet").click()
    at.run()


def add_task(at: AppTest, title: str, minutes: int = 15) -> None:
    labelled(at.sidebar.text_input, "Task").input(title)
    labelled(at.sidebar.number_input, "Minutes").set_value(minutes)
    labelled(at.sidebar.button, "Add task").click()
    at.run()


def test_empty_state_without_key(app):
    assert not app.exception
    assert "Nothing planned yet" in app.info[0].value
    assert labelled(app.button, "Plan it for me").disabled
    assert any("GEMINI_API_KEY" in caption.value for caption in app.caption)


def test_added_task_appears_in_todays_plan(app):
    add_pet(app, "Mochi")
    add_task(app, "Walk", 30)
    assert not app.exception
    assert [box.label for box in app.main.checkbox] == ["**08:00–08:30** &nbsp; Walk — Mochi · 30 min"]


def test_ticking_moves_task_to_done_and_queues_next(app):
    add_pet(app, "Mochi")
    add_task(app, "Walk")
    app.main.checkbox[0].check()
    app.run()
    assert not app.exception
    assert [box.label for box in app.main.checkbox] == ["~~Walk — Mochi · 15 min~~"]
    assert "All done for today" in app.success[0].value
    assert any(exp.label == "Coming up (1)" for exp in app.expander)

    app.main.checkbox[0].uncheck()
    app.run()
    assert [box.value for box in app.main.checkbox] == [False]
    assert not any(exp.label.startswith("Coming up") for exp in app.expander)


def test_task_over_budget_is_reported(app):
    add_pet(app, "Mochi")
    add_task(app, "Walk", 30)
    app.sidebar.number_input(key="available_minutes").set_value(10)
    app.run()
    assert "Didn't fit in your 10 minutes" in app.warning[0].value
    assert "Walk (Mochi, 30 min)" in app.warning[0].value


def test_ai_run_fills_in_the_app(app, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    script = [
        call("set_time_budget", minutes=90),
        call("add_pet", name="Mochi", animal_type="dog"),
        call("add_task", pet_name="Mochi", title="Morning walk", duration_minutes=40,
             priority="high", frequency="daily"),
        call("build_plan"),
        call("finish", explanation="A long walk for a husky."),
    ]
    monkeypatch.setattr(agent, "GeminiClient", lambda: FakeLLM(script))
    app.run()
    app.text_area[0].input("Mochi is a husky, I have 90 minutes")
    labelled(app.button, "Plan it for me").click()
    app.run()

    assert not app.exception
    assert [pet.name for pet in app.session_state.pets] == ["Mochi"]
    assert app.sidebar.number_input(key="available_minutes").value == 90
    assert any("Added 1 pet(s) and 1 task(s)" in m.value for m in app.markdown)
    assert any(exp.label == "Why this plan?" for exp in app.expander)
    assert [box.label for box in app.main.checkbox] == ["**08:00–08:40** &nbsp; Morning walk — Mochi · 40 min"]
    assert not app.warning  # the plan fits, so the post-check has nothing to say


def test_quick_add_uses_the_trained_model(app):
    add_pet(app, "Luna")
    labelled(app.sidebar.text_input, "Quick add").input("give kidney meds at 8am")
    labelled(app.sidebar.button, "Add").click()
    app.run()
    assert not app.exception
    task = app.session_state.pets[0].tasks[0]
    assert (task.title, task.priority.name, task.frequency, task.required_start) == (
        "Give kidney meds", "HIGH", "daily", 480,
    )
    assert "📌" in app.main.checkbox[0].label


def plan(app: AppTest, request: str) -> None:
    """Type a request into Plan with AI and press the button."""
    app.text_area[0].input(request)
    labelled(app.button, "Plan it for me").click()
    app.run()
