import math
import re


def transcript_segments_overlap(segment_one: str, segment_two: str) -> int:
    n = min(len(segment_one), len(segment_two))
    i = 0
    while i < n and segment_one[-(i + 1)] == segment_two[-(i + 1)]:
        i += 1
    return i


def normalise_text(text: str) -> list[str]:
    """
    Takes a string, converts it to lowercase, removes punctuation,
    and splits it into a list of words.
    """
    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove punctuation.
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Split into words
    words = [word for word in text.split() if word]

    return words


def find_normalised_word_overlap(text1: str, text2: str) -> int:
    """
    Finds the number of overlapping words between the end of text1 and the
    start of text2, after normalizing them.
    """
    # Normalize both input texts into lists of words
    words1 = normalise_text(text1)
    words2 = normalise_text(text2)

    n = min(len(words1), len(words2))
    i = 0
    while i < n and words1[-(i + 1)] == words2[-(i + 1)]:
        i += 1
    return i


def plausible_overlap(
    overlap: int, expected_overlap: int, min_percent: float, max_percent: float
) -> bool:
    """
    Check that word overlap length is in a sensible range so it makes sense to stitch transcript segments.
    """
    if abs(min_percent) >= 1 or min_percent >= max_percent:
        raise ValueError(
            "Min should be between 0 and 1 and max should be higher than min"
        )
    min_overlap = math.floor(expected_overlap * min_percent)
    max_overlap = math.ceil(expected_overlap * max_percent)
    return min_overlap <= overlap <= max_overlap
