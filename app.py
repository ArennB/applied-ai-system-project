"""PawPal+ Streamlit app.

Layout:
    Sidebar  - your day (name, time available), pets, quick add (trained task
               model) or a detailed task form, how it works
    Main     - Plan with AI (agent), Ask the care guide (RAG Q&A), then today's
               live plan with tick-to-complete, what didn't fit, done today,
               coming up, and a task editor

The plan is rebuilt from the scheduler on every rerun, so there is exactly one
plan on screen and it is always current.

Guardrails: every text box has a length cap, anything sent to the AI is first
checked by pawpal_guardrails (cleaned, injection-screened, emergency-flagged),
and each session may make at most AI_CALLS_PER_WINDOW AI calls per window.
Activity is logged to logs/pawpal.log (see pawpal_logging).
"""

import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv

# Read GEMINI_API_KEY (and other settings) from a git-ignored .env file next to
# this script. Variables already set in the shell win over the file. The test
# suite sets PAWPAL_SKIP_DOTENV so a developer's real key never leaks into tests.
if not os.environ.get("PAWPAL_SKIP_DOTENV"):
    load_dotenv()

from pawpal_agent import GeminiClient, LLMError, PlanningAgent, gemini_api_key
from pawpal_guardrails import (
    MAX_QUESTION_CHARS,
    MAX_QUICK_ADD_CHARS,
    MAX_REQUEST_CHARS,
    URGENT_NOTICE,
    GuardrailError,
    RateLimiter,
    check_input,
)
from pawpal_logging import get_logger, preview, setup_logging
from pawpal_qa import CareGuideQA, QAResult
from pawpal_rag import KnowledgeBase
from pawpal_system import Owner, Pet, Priority, Scheduler, Task
from pawpal_ui import MAX_TITLE_CHARS, apply_task_edits, editor_rows, task_label, to_minutes

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")
setup_logging()
log = get_logger("app")

# Prefix for the per-task "done" checkbox keys, so they can be cleared in bulk.
DONE_PREFIX = "done_"
SPECIES_ICON = {"dog": "🐕", "cat": "🐈", "other": "🐾"}
PRIORITIES = ["low", "medium", "high"]
FREQUENCIES = ["once", "daily", "weekly"]
# Maps the status filter onto filter_tasks' is_completed, where None means "any".
STATUS_FILTERS = {"All": None, "To do": False, "Done": True}
# Per-session cap on Gemini calls (planner and Ask combined).
AI_CALLS_PER_WINDOW, AI_WINDOW_SECONDS = 10, 300
MAX_NAME_CHARS = 40


def done_key(task: Task) -> str:
    """Checkbox key for a task, tied to the object rather than its list position.

    Completing a task inserts a new one into the list, which would shift
    positional keys and move tick marks onto the wrong rows. The same key is
    used whether the task is shown in the plan or under Done today, so ticking
    it in one place and unticking it in the other stay in step.
    """
    return f"{DONE_PREFIX}{id(task)}"


def clear_done_keys() -> None:
    """Forget every checkbox state, e.g. after tasks were replaced by new objects."""
    for key in [k for k in st.session_state if str(k).startswith(DONE_PREFIX)]:
        del st.session_state[key]


@st.cache_resource
def load_knowledge() -> KnowledgeBase:
    """The care guide in knowledge/, indexed once per server process."""
    return KnowledgeBase.from_directory()


@st.cache_resource
def load_task_model():
    """The trained task model, or None if it cannot be built here.

    Trained from data/task_dataset.csv on first use (well under a second) and
    shared across sessions. Missing scikit-learn or a missing dataset turns the
    model-backed features off rather than breaking the app.
    """
    try:
        from pawpal_model import TaskModel

        return TaskModel.train()
    except (ImportError, OSError, ValueError) as exc:
        log.warning("task model unavailable, quick add and second opinions off: %s", exc)
        return None


def show_passage(number: int, passage) -> None:
    """One care-guide passage, numbered to match the answer's [n] citations."""
    st.markdown(f"**[{number}]** *{passage.cite()}*")
    st.caption(passage.text)


def ai_call_allowed(what: str) -> bool:
    """Spend one of this session's AI calls, or explain why it can't."""
    limiter: RateLimiter = st.session_state.ai_limiter
    if limiter.try_acquire(what):
        return True
    st.warning(
        f"You've used PawPal's AI {limiter.max_calls} times in the last few minutes."
        f" Try again in {int(limiter.retry_after()) + 1} seconds."
    )
    return False


def flag_if_urgent(check) -> None:
    """Remember that the user may be describing an emergency, for the banner."""
    if check.urgent:
        st.session_state.urgent = True


def show_qa(result: QAResult) -> None:
    """Render an Ask the care guide result according to how grounded it is."""
    if result.status == "blocked":
        st.warning(result.answer)
    elif result.status == "not_covered":
        st.info(
            "The care guide doesn't cover that, so PawPal won't guess."
            " For anything about your pet's health, ask your vet."
        )
    elif result.status == "retrieval_only":
        st.caption(
            "Here's what the care guide says."
            + ("" if gemini_api_key() else " Set GEMINI_API_KEY to get a written answer as well.")
        )
        for number, passage in enumerate(result.passages, start=1):
            show_passage(number, passage)
    elif result.status == "ungrounded":
        for warning in result.warnings or ["This answer may not come from the care guide."]:
            st.warning(f"{warning} The passages are below.")
        st.write(result.answer)
        for number, passage in enumerate(result.passages, start=1):
            show_passage(number, passage)
    else:
        st.write(result.answer)
        st.markdown("**Sources**")
        for number, passage in enumerate(result.passages, start=1):
            if any(passage is cited for cited in result.citations):
                show_passage(number, passage)


# Pets accumulate across interactions, so they live in session_state to survive
# Streamlit's rerun on every click. Tasks ride along inside each Pet.
st.session_state.setdefault("pets", [])
# Bumped whenever the task editor's changes are applied, which gives the editor
# a fresh key so its stored row edits cannot be replayed onto shifted rows.
st.session_state.setdefault("editor_version", 0)
st.session_state.setdefault("ai_limiter", RateLimiter(AI_CALLS_PER_WINDOW, AI_WINDOW_SECONDS))
pets: list[Pet] = st.session_state.pets
task_model = load_task_model()

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("👤 Your day")
    owner_name = st.text_input("Your name", placeholder="e.g. Jordan", max_chars=MAX_NAME_CHARS)
    # Keyed so the AI planner can set it. A widget's value cannot be changed
    # after it is drawn in a run, so the planner leaves the new value in
    # pending_budget and reruns, and it is applied here before the widget exists.
    if "pending_budget" in st.session_state:
        st.session_state.available_minutes = st.session_state.pop("pending_budget")
    st.session_state.setdefault("available_minutes", 60)
    available_minutes = int(
        st.number_input(
            "Minutes for pet care today", min_value=5, max_value=600, step=5,
            key="available_minutes",
        )
    )

    st.header("🐶 Pets")
    for index, pet in enumerate(pets):
        col_pet, col_remove = st.columns([5, 1], vertical_alignment="center")
        outstanding = sum(not task.is_completed for task in pet.tasks)
        col_pet.markdown(
            f"{SPECIES_ICON.get(pet.animal_type, '🐾')} **{pet.name}** · {outstanding} to do"
        )
        if col_remove.button("🗑", key=f"remove_pet_{index}_{pet.name}", help=f"Remove {pet.name}"):
            pets.remove(pet)
            log.info("pet removed: %s", pet.name)
            st.rerun()

    with st.form("add_pet", clear_on_submit=True, border=False):
        new_pet_name = st.text_input("Pet name", placeholder="e.g. Mochi", max_chars=MAX_NAME_CHARS)
        new_species = st.selectbox("Species", list(SPECIES_ICON))
        if st.form_submit_button("Add pet"):
            name = new_pet_name.strip()
            if not name:
                st.warning("Give the pet a name first.")
            elif any(pet.name.lower() == name.lower() for pet in pets):
                st.warning(f"{name} is already on the list.")
            else:
                pets.append(Pet(name, new_species))
                log.info("pet added: %s (%s)", name, new_species)
                st.rerun()

    st.header("➕ Add a task")
    if not pets:
        st.caption("Add a pet first.")
    else:
        if task_model is None:
            st.caption("Quick add needs scikit-learn: pip install -r requirements.txt")
        else:
            with st.form("quick_add", clear_on_submit=True, border=False):
                quick_pet = st.selectbox("For", [pet.name for pet in pets], key="quick_pet")
                quick_text = st.text_input(
                    "Quick add", placeholder="e.g. evening walk 30 min", max_chars=MAX_QUICK_ADD_CHARS,
                    help=(
                        "Type it the way you'd say it. PawPal's trained task model"
                        " picks the priority, how often it repeats and, unless you"
                        " say, how long it takes. Add 'at 7pm' for a set time."
                    ),
                )
                if st.form_submit_button("Add"):
                    # No language model sees this text, so only clean and cap it.
                    check = check_input(quick_text, MAX_QUICK_ADD_CHARS, "task", screen_injection=False)
                    if not check.allowed:
                        st.warning(check.reason)
                    else:
                        # Imported here: pawpal_model needs scikit-learn, which is optional.
                        from pawpal_model import parse_quick_add

                        parsed = parse_quick_add(check.text)
                        target = next(pet for pet in pets if pet.name == quick_pet)
                        guess = task_model.predict(parsed.title, target.animal_type)
                        minutes = parsed.minutes or guess.minutes
                        target.add_task(
                            Task(
                                parsed.title,
                                minutes,
                                guess.priority,
                                frequency=guess.frequency,
                                required_start=parsed.fixed_start,
                            )
                        )
                        log.info(
                            "quick add %r for %s -> %s, %d min, %s (confidence %s)",
                            preview(check.text), target.name, guess.priority.name.lower(),
                            minutes, guess.frequency, guess.confidence,
                        )
                        st.toast(
                            f"Added {parsed.title} for {target.name}:"
                            f" {guess.priority.name.lower()} · {minutes} min ·"
                            f" {guess.frequency}. Change it under Manage tasks.",
                            icon="⚡",
                        )
                        st.rerun()

        with st.expander("Or fill in every detail", expanded=task_model is None), st.form(
            "add_task", clear_on_submit=True, border=False
        ):
            task_pet = st.selectbox("For", [pet.name for pet in pets])
            task_title = st.text_input("Task", placeholder="e.g. Morning walk", max_chars=MAX_TITLE_CHARS)
            task_minutes = st.number_input("Minutes", min_value=1, max_value=240, value=15)
            task_priority = st.segmented_control(
                "Priority", PRIORITIES, default="medium", selection_mode="single"
            )
            task_frequency = st.segmented_control(
                "Repeats", FREQUENCIES, default="daily", selection_mode="single"
            )
            task_time = st.time_input(
                "At a set time? (optional)", value=None, step=300,
                help="Only for things that can't move, like a vet visit or timed medication.",
            )
            if st.form_submit_button("Add task"):
                if not task_title.strip():
                    st.warning("Give the task a name first.")
                else:
                    target = next(pet for pet in pets if pet.name == task_pet)
                    target.add_task(
                        Task(
                            task_title.strip(),
                            int(task_minutes),
                            Priority[(task_priority or "medium").upper()],
                            frequency=task_frequency or "once",
                            required_start=to_minutes(task_time),
                        )
                    )
                    log.info("task added: %r for %s", task_title.strip(), target.name)
                    st.toast(f"Added {task_title.strip()} for {target.name}.", icon="✅")
                    st.rerun()

    with st.expander("ℹ️ How PawPal plans your day"):
        st.markdown(
            """
- Tasks at a **set time** (📌) are placed first, at that time.
- Everything else goes in **highest priority first**, shorter tasks first on
  ties, into the first free slot from 08:00.
- If a task won't fit in your minutes, it's listed under **didn't fit** and the
  next one is tried.
- Ticking off a **daily** or **weekly** task schedules the next one automatically.
"""
        )

# --- Shared state for the main page --------------------------------------------

owner = Owner(owner_name.strip() or "the owner", available_minutes=available_minutes)
for pet in pets:
    owner.add_pet(pet)
scheduler = Scheduler()
today = date.today()

# Apply last run's tick changes before drawing anything, so the plan reflects
# them now rather than one click late. Completing goes through the Pet, which
# also queues the next occurrence of a recurring task; unticking removes it.
for pet, task in owner.all_tasks():
    ticked = st.session_state.get(done_key(task))
    if ticked and not task.is_completed:
        follow_on = pet.complete_task(task)
        log.info("task done: %r for %s", task.title, pet.name)
        message = f"Done: {task.title} for {pet.name}."
        if follow_on is not None:
            message += f" Next one is due {follow_on.due_date:%a %d %b}."
        st.toast(message, icon="🎉")
    elif ticked is False and task.is_completed:
        pet.uncomplete_task(task)

# --- Main: header and AI planner ----------------------------------------------------

st.title("🐾 PawPal+")
st.caption("Tell PawPal about your pets and your day, then tick things off as you go.")

if st.session_state.get("urgent"):
    st.error(URGENT_NOTICE, icon="🚨")
    if st.button("Got it", key="dismiss_urgent"):
        del st.session_state.urgent
        st.rerun()

api_key_present = gemini_api_key() is not None
with st.expander("✨ Plan with AI", expanded=not pets):
    ai_request = st.text_area(
        "Describe your pets and your day",
        placeholder=(
            "e.g. Mochi is a 2-year-old husky and Luna is a 14-year-old cat with"
            " kidney disease who gets meds at 08:00. I have 75 minutes today and"
            " Luna has a vet visit at 10:00."
        ),
        label_visibility="collapsed",
        max_chars=MAX_REQUEST_CHARS,
    )
    if not api_key_present:
        st.caption(
            "To use this, add GEMINI_API_KEY=your-key to a .env file in the project folder and restart the app."
            " You can still add pets and tasks in the sidebar."
        )
    if st.button("Plan it for me", type="primary", disabled=not api_key_present):
        check = check_input(ai_request, MAX_REQUEST_CHARS, "request")
        flag_if_urgent(check)
        if not check.text:
            st.warning("Describe your pets and your day first.")
        elif not check.allowed:
            st.warning(check.reason)
        elif ai_call_allowed("planner"):
            pets_before = len(pets)
            tasks_before = sum(len(pet.tasks) for pet in pets)
            with st.spinner("Reading the care guide and building your plan..."):
                try:
                    agent = PlanningAgent(GeminiClient(), load_knowledge(), task_model=task_model)
                    result = agent.run(check.text, owner.name, available_minutes, pets)
                except GuardrailError as exc:
                    st.warning(str(exc))
                except LLMError as exc:
                    # The agent works on a copy, so a failed run changes nothing.
                    log.error("planner failed: %s", exc)
                    st.error(str(exc))
                else:
                    st.session_state.pets = result.pets
                    st.session_state.ai_result = result
                    st.session_state.ai_summary = (
                        f"Added {len(result.pets) - pets_before} pet(s) and"
                        f" {sum(len(p.tasks) for p in result.pets) - tasks_before} task(s)."
                    )
                    if result.available_minutes != available_minutes:
                        st.session_state.pending_budget = result.available_minutes
                    # Every task is a new object now; drop the old checkbox keys
                    # so a recycled object id cannot inherit someone else's tick.
                    clear_done_keys()
                    st.rerun()

ai_result = st.session_state.get("ai_result")
if ai_result is not None:
    with st.container(border=True):
        col_summary, col_hide = st.columns([6, 1], vertical_alignment="center")
        col_summary.markdown(f"**✨ {st.session_state.get('ai_summary', 'Plan ready.')}**")
        if col_hide.button("Hide", type="tertiary"):
            del st.session_state.ai_result
            st.rerun()
        if ai_result.explanation:
            st.write(ai_result.explanation)
        # Skipped tasks and clashes are shown live in Today below, so only the
        # issues about the run itself are repeated here.
        for issue in ai_result.process_issues:
            st.warning(issue)
        if ai_result.second_opinions:
            count = len(ai_result.second_opinions)
            st.caption(
                f"🔎 PawPal's task model would set {count} priorit{'y' if count == 1 else 'ies'}"
                " differently. See Why this plan?"
            )
        with st.expander("Why this plan?"):
            st.markdown("**From the care guide**")
            if not ai_result.citations:
                st.caption("The assistant didn't look anything up.")
            for passage in ai_result.citations:
                st.markdown(f"*{passage.cite()}*")
                st.caption(passage.text)
            st.markdown("**What the assistant did**")
            st.dataframe(
                [
                    {"Step": s.step, "Action": s.tool, "Input": str(s.args), "Result": s.outcome}
                    for s in ai_result.trace
                ],
                hide_index=True,
            )
            if ai_result.second_opinions:
                st.markdown("**Second opinion from the task model**")
                st.caption(
                    "A classifier trained on labelled pet care tasks checked each task"
                    " the assistant added. These are the ones where it confidently"
                    " disagrees on priority. It's advice only; change them under"
                    " Manage tasks if you agree."
                )
                st.dataframe(
                    [
                        {
                            "Task": o.task,
                            "Pet": o.pet,
                            "Assistant": o.agent_priority,
                            "Task model": o.model_priority,
                            "Model confidence": f"{o.confidence:.0%}",
                        }
                        for o in ai_result.second_opinions
                    ],
                    hide_index=True,
                )

with st.expander("📖 Ask the care guide"):
    with st.form("ask_guide", border=False):
        question = st.text_input(
            "Question",
            placeholder="e.g. How often should I brush a long-haired cat?",
            label_visibility="collapsed",
            max_chars=MAX_QUESTION_CHARS,
        )
        asked = st.form_submit_button("Ask")
    if asked and question.strip():
        check = check_input(question, MAX_QUESTION_CHARS, "question")
        flag_if_urgent(check)
        # Only spend an AI call if the question passes the input guardrails;
        # otherwise ask() answers it (blocked, or passages only) without one.
        use_model = api_key_present and check.allowed and ai_call_allowed("question")
        with st.spinner("Looking it up..."):
            try:
                qa = CareGuideQA(load_knowledge(), GeminiClient() if use_model else None)
                st.session_state.qa_result = qa.ask(check.text)
            except LLMError as exc:
                log.error("question failed: %s", exc)
                st.error(str(exc))
        if check.urgent:
            st.rerun()  # show the emergency banner at the top straight away
    if "qa_result" in st.session_state:
        show_qa(st.session_state.qa_result)

# --- Main: today's live plan ------------------------------------------------------

st.subheader("📅 Today")

if not owner.all_tasks():
    st.info(
        "Nothing planned yet. Describe your day above, or add a pet and some tasks"
        " in the sidebar to get started."
    )
else:
    schedule = scheduler.build_plan(owner, on=today)
    done = owner.filter_tasks(is_completed=True)
    to_do_today = len(schedule.entries) + len(schedule.skipped)
    st.caption(
        f"{schedule.total_minutes} of {available_minutes} min planned"
        f" · {len(done)} of {len(done) + to_do_today} done"
    )

    if not schedule.entries and not schedule.skipped:
        st.success("All done for today! 🎉")

    for entry in sorted(schedule.entries, key=lambda e: e.start_minutes):
        label = f"**{entry.start_time}–{entry.end_time}** &nbsp; {task_label(entry.task, entry.pet)}"
        if entry.task.is_pinned:
            label += " 📌"
        if entry.task.due_date < today:
            label += " · *overdue*"
        st.checkbox(label, value=False, key=done_key(entry.task))

    if schedule.skipped:
        skipped = ", ".join(
            f"{task.title} ({pet.name}, {task.duration_minutes} min)"
            for pet, task in scheduler.sort_tasks(schedule.skipped)
        )
        st.warning(
            f"**Didn't fit in your {available_minutes} minutes:** {skipped}. Add more"
            " time in the sidebar, or shorten or remove something under Manage tasks."
        )
    for warning in schedule.warnings:
        st.error(f"**Time clash:** {warning}. Change one of the set times under Manage tasks.")

    if done:
        st.markdown("**✅ Done today**")
        for pet, task in done:
            st.checkbox(f"~~{task_label(task, pet)}~~", value=True, key=done_key(task))

    upcoming = scheduler.upcoming_tasks(owner, after=today)
    if upcoming:
        with st.expander(f"Coming up ({len(upcoming)})"):
            for pet, task in upcoming:
                st.markdown(f"{task.due_date:%a %d %b} · {task_label(task, pet)} · {task.frequency}")

    with st.expander("Manage tasks"):
        col_pet, col_status = st.columns(2)
        pet_filter = col_pet.selectbox("Pet", ["All pets"] + [pet.name for pet in pets])
        status_filter = col_status.selectbox("Status", list(STATUS_FILTERS))
        rows = scheduler.sort_tasks(
            owner.filter_tasks(
                pet_name=None if pet_filter == "All pets" else pet_filter,
                is_completed=STATUS_FILTERS[status_filter],
            )
        )
        if not rows:
            st.caption("No tasks match.")
        else:
            original = editor_rows(rows)
            edited = st.data_editor(
                original,
                key=f"editor_{st.session_state.editor_version}_{pet_filter}_{status_filter}",
                hide_index=True,
                disabled=["Pet", "Status"],
                column_config={
                    "Task": st.column_config.TextColumn(max_chars=MAX_TITLE_CHARS, required=True),
                    "Minutes": st.column_config.NumberColumn(min_value=1, max_value=240, step=1, required=True),
                    "Priority": st.column_config.SelectboxColumn(options=PRIORITIES, required=True),
                    "Repeats": st.column_config.SelectboxColumn(options=FREQUENCIES, required=True),
                    "Set time": st.column_config.TimeColumn(format="HH:mm", step=300),
                    "Remove": st.column_config.CheckboxColumn(help="Tick to delete this task"),
                },
            )
            st.caption("Edit a cell to change a task, or tick Remove to delete it.")

            if hasattr(edited, "to_dict"):  # a DataFrame, on some Streamlit versions
                edited = edited.to_dict("records")

            changed = apply_task_edits(rows, edited)
            if changed:
                # A fresh editor key, so the stored edits are not re-applied to
                # rows that have moved (a removed task shifts everything up).
                st.session_state.editor_version += 1
                st.rerun()
