"""Small text helpers shared by domains and renderers."""

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def join_nl(phrases: list[str], conj: str) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f" {conj} " + phrases[-1]


def number_word(k: int) -> str:
    return NUMBER_WORDS.get(k, str(k))
