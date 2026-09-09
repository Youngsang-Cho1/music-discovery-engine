"""
Tests for data preprocessing utilities (src/audio_utils.py).

Validates the join-key generation logic that must be stable across the
entire pipeline (audio download → feature extraction → dataset merge).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_utils import get_safe_name


def test_safe_name_strips_special_characters():
    """Only alphanumeric characters and spaces should survive."""
    result = get_safe_name("Hello, World!", "Artist & More")
    for char in result:
        assert char.isalpha() or char.isdigit() or char == ' ', \
            f"Unexpected character '{char}' in safe name"


def test_safe_name_is_deterministic():
    """Same input must always produce the same join key (pipeline stability)."""
    name1 = get_safe_name("Shape of You", "Ed Sheeran")
    name2 = get_safe_name("Shape of You", "Ed Sheeran")
    assert name1 == name2


def test_safe_name_no_trailing_whitespace():
    """Safe names must not end in whitespace (causes silent join mismatches)."""
    result = get_safe_name("Song!", "Artist!")
    assert result == result.rstrip(), f"Trailing whitespace found: '{result}'"


def test_safe_name_different_songs_differ():
    """Different songs must produce different keys."""
    key1 = get_safe_name("Blinding Lights", "The Weeknd")
    key2 = get_safe_name("Save Your Tears", "The Weeknd")
    assert key1 != key2


def test_safe_name_case_preserved():
    """Case is preserved — join keys are case-sensitive by convention."""
    key = get_safe_name("YMCA", "Village People")
    assert "YMCA" in key


def test_safe_name_handles_unicode():
    """Non-ASCII characters are stripped without raising exceptions."""
    result = get_safe_name("Café del Mar", "Energy 52")
    assert isinstance(result, str)
    assert len(result) > 0


if __name__ == '__main__':
    test_safe_name_strips_special_characters()
    test_safe_name_is_deterministic()
    test_safe_name_no_trailing_whitespace()
    test_safe_name_different_songs_differ()
    test_safe_name_case_preserved()
    test_safe_name_handles_unicode()
    print("All preprocessing tests passed.")
