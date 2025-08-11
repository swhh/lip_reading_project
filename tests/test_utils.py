import pytest

from lip_reading_project.utils import find_normalised_word_overlap, normalise_text, transcript_segments_overlap




@pytest.mark.parametrize("segment1, segment2, expected", [
    ("hello i am writing", "i am writing", len("i am writing")),
    ("this", "this", len("this")),
    ("this", "that", 0),
    ("", "this", 0),
])
def test_transcript_segments_overlap(segment1, segment2, expected):
    assert transcript_segments_overlap(segment1, segment2) == expected


@pytest.mark.parametrize("segment1, segment2, expected", [
    ("hello, i am writing", "i am writing", 3),
    ("this,", "this", 1),
    ("this", "that", 0),
    ("", "this", 0),
])
def test_find_normalised_word_overlap(segment1, segment2, expected):
    assert find_normalised_word_overlap(segment1, segment2) == expected






    


