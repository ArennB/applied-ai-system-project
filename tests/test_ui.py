"""Tests for the pure UI helpers in ui_helpers, chiefly the task editor round trip."""

from datetime import time

from scheduler import Pet, Priority, Task
from ui_helpers import apply_task_edits, cell_minutes, editor_rows, to_minutes, to_time


def pet_with(*tasks: Task) -> Pet:
    return Pet("Mochi", "dog", list(tasks))


def pairs(pet: Pet):
    return [(pet, task) for pet_ in [pet] for task in pet_.tasks]


def test_time_conversions_round_trip():
    assert to_minutes(time(9, 30)) == 570
    assert to_time(570) == time(9, 30)
    assert to_minutes(None) is None and to_time(None) is None


def test_cell_minutes_accepts_time_or_string():
    assert cell_minutes(time(10, 0)) == 600
    assert cell_minutes("10:00:00") == 600
    assert cell_minutes("10:00") == 600
    assert cell_minutes(None) is None
    assert cell_minutes("") is None
    assert cell_minutes(float("nan")) is None


def test_unchanged_rows_report_no_change():
    pet = pet_with(Task("Walk", 30, Priority.HIGH, required_start=480))
    rows = pairs(pet)
    assert apply_task_edits(rows, editor_rows(rows)) is False


def test_same_values_in_other_types_are_not_a_change():
    pet = pet_with(Task("Walk", 30, Priority.HIGH, required_start=480))
    rows = pairs(pet)
    edited = editor_rows(rows)
    edited[0].update({"Minutes": 30.0, "Set time": "08:00:00"})
    assert apply_task_edits(rows, edited) is False


def test_edits_are_written_back():
    pet = pet_with(Task("Walk", 30, Priority.HIGH))
    rows = pairs(pet)
    edited = editor_rows(rows)
    edited[0].update(
        {"Task": "Long walk", "Minutes": 45, "Priority": "low", "Repeats": "weekly", "Set time": time(7, 0)}
    )
    assert apply_task_edits(rows, edited) is True
    task = pet.tasks[0]
    assert (task.title, task.duration_minutes, task.priority, task.frequency, task.required_start) == (
        "Long walk", 45, Priority.LOW, "weekly", 420,
    )


def test_clearing_set_time_makes_task_flexible():
    pet = pet_with(Task("Vet", 30, Priority.HIGH, required_start=600))
    rows = pairs(pet)
    edited = editor_rows(rows)
    edited[0]["Set time"] = None
    assert apply_task_edits(rows, edited) is True
    assert pet.tasks[0].required_start is None


def test_bad_cells_keep_old_values():
    pet = pet_with(Task("Walk", 30, Priority.HIGH))
    rows = pairs(pet)
    edited = editor_rows(rows)
    edited[0].update({"Task": "  ", "Minutes": 999, "Priority": "urgent", "Repeats": "hourly"})
    assert apply_task_edits(rows, edited) is False
    task = pet.tasks[0]
    assert (task.title, task.duration_minutes, task.priority, task.frequency) == (
        "Walk", 30, Priority.HIGH, "once",
    )


def test_remove_deletes_exactly_that_task():
    walk, walk_twin = Task("Walk", 30, Priority.HIGH), Task("Walk", 30, Priority.HIGH)
    pet = pet_with(walk, walk_twin)
    rows = pairs(pet)
    edited = editor_rows(rows)
    edited[1]["Remove"] = True
    assert apply_task_edits(rows, edited) is True
    # The twins compare equal as dataclasses, so removal must go by identity.
    assert len(pet.tasks) == 1 and pet.tasks[0] is walk
