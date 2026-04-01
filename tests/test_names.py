"""Tests for four-word session ID generator."""

import pytest

from psoul.names import ADJECTIVES, OBJECT_NOUNS, SUBJECT_NOUNS, VERBS, generate_session_id
from psoul.session import validate_session_id


@pytest.mark.parametrize(
    ("name", "word_list"),
    [
        ("ADJECTIVES", ADJECTIVES),
        ("SUBJECT_NOUNS", SUBJECT_NOUNS),
        ("OBJECT_NOUNS", OBJECT_NOUNS),
        ("VERBS", VERBS),
    ],
)
def test_word_list_has_64_unique_entries(name: str, word_list: tuple[str, ...]) -> None:
    assert len(word_list) == 64, f"{name} has {len(word_list)} entries, expected 64"
    assert len(set(word_list)) == 64, f"{name} contains duplicates"


def test_generated_id_segments_from_correct_lists() -> None:
    session_id = generate_session_id()
    adj, subj, verb, obj = session_id.split("-", maxsplit=3)
    assert adj in ADJECTIVES
    assert subj in SUBJECT_NOUNS
    assert verb in VERBS
    assert obj in OBJECT_NOUNS


def test_deterministic_with_patched_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_choice(seq: tuple[str, ...]) -> str:
        calls.append(seq)
        return seq[0]

    monkeypatch.setattr("psoul.names.secrets.choice", fake_choice)
    session_id = generate_session_id()
    assert session_id == f"{ADJECTIVES[0]}-{SUBJECT_NOUNS[0]}-{VERBS[0]}-{OBJECT_NOUNS[0]}"
    assert len(calls) == 4


def test_generated_id_passes_validation() -> None:
    session_id = generate_session_id()
    assert validate_session_id(session_id) == session_id
