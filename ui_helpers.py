"""Pure helpers behind the Streamlit app, kept free of Streamlit so they can be
unit-tested directly.

Most importantly this holds the task editor's round trip: editor_rows() turns
(pet, task) pairs into table rows, and apply_task_edits() writes whatever the
user changed in that table back onto the Task objects.
"""

from datetime import time

from scheduler import Pet, Priority, Task, parse_time

MIN_MINUTES, MAX_MINUTES = 1, 240
MAX_TITLE_CHARS = 80
FREQUENCIES = ("once", "daily", "weekly")


def to_minutes(value: time | None) -> int | None:
    """A time picker's value as minutes since midnight, or None for "any time"."""
    return None if value is None else value.hour * 60 + value.minute


def to_time(minutes: int | None) -> time | None:
    """Minutes since midnight as a time for a picker, or None."""
    return None if minutes is None else time(minutes // 60, minutes % 60)


def cell_minutes(value) -> int | None:
    """A "Set time" cell from the task editor as minutes since midnight, or None.

    The editor may hand back a time object or an "HH:MM[:SS]" string depending
    on the Streamlit version, so both are accepted; anything else means unset.
    """
    if isinstance(value, time):
        return to_minutes(value)
    if isinstance(value, str) and value.strip():
        return parse_time(":".join(value.strip().split(":")[:2]))
    return None


def task_label(task: Task, pet: Pet) -> str:
    """One-line description of a task for lists: "Walk — Mochi · 30 min"."""
    return f"{task.title} — {pet.name} · {task.duration_minutes} min"


def editor_rows(pet_tasks: list[tuple[Pet, Task]]) -> list[dict]:
    """Rows for the task editor, one per (pet, task), in the order given."""
    return [
        {
            "Task": task.title,
            "Pet": pet.name,
            "Minutes": task.duration_minutes,
            "Priority": task.priority.name.lower(),
            "Repeats": task.frequency,
            "Set time": to_time(task.required_start),
            "Status": "Done" if task.is_completed else "To do",
            "Remove": False,
        }
        for pet, task in pet_tasks
    ]


def apply_task_edits(pet_tasks: list[tuple[Pet, Task]], edited: list[dict]) -> bool:
    """Write the editor's rows back onto their tasks; True if anything changed.

    Rows line up with pet_tasks by position. Each row's values are parsed and
    compared against the task itself rather than against the original row, so
    a cell that merely comes back as a different type (a time as a string, an
    int as a float) does not count as a change and cannot trigger a rerun loop.

    Bad cells are ignored rather than applied: a blank or over-long title keeps
    the old one, and minutes, priority or repeats outside the allowed values keep theirs.
    A ticked Remove deletes the task from its pet.
    """
    changed = False
    for (pet, task), row in zip(pet_tasks, edited):
        if row.get("Remove"):
            pet.tasks = [item for item in pet.tasks if item is not task]
            changed = True
            continue
        updates = {
            "title": _title_or(row.get("Task"), task.title),
            "duration_minutes": _minutes_or(row.get("Minutes"), task.duration_minutes),
            "priority": _priority_or(row.get("Priority"), task.priority),
            "frequency": row.get("Repeats") if row.get("Repeats") in FREQUENCIES else task.frequency,
            "required_start": cell_minutes(row.get("Set time")),
        }
        for attribute, value in updates.items():
            if getattr(task, attribute) != value:
                setattr(task, attribute, value)
                changed = True
    return changed


def _title_or(value, fallback: str) -> str:
    title = " ".join(str(value or "").split())
    return title if 0 < len(title) <= MAX_TITLE_CHARS else fallback


def _minutes_or(value, fallback: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not number.is_integer() or not MIN_MINUTES <= number <= MAX_MINUTES:
        return fallback
    return int(number)


def _priority_or(value, fallback: Priority) -> Priority:
    try:
        return Priority[str(value).strip().upper()]
    except KeyError:
        return fallback
