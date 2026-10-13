import pytest

from nsg_sim.prompts.build import build_prompt, check_no_leak
from nsg_sim.prompts.parse import parse_answer


@pytest.mark.parametrize("text,expected", [
    ("[ANSWER]\nYES\n\n[CONFIDENCE]\nHIGH", "YES"),
    ("[ANSWER] NO", "NO"),
    ("**ANSWER**\nNO\n**CONFIDENCE** LOW", "NO"),           # llama3
    ("**Answer:** yes", "YES"),
    ("## Answer\nNo.", "NO"),
    ("Answer: YES", "YES"),
    ("<think>[ANSWER] NO</think>\n[ANSWER]\nYES", "YES"),    # reasoning block dropped
    ("<think>[ANSWER] NO", None),                           # unterminated reasoning, no answer
    ("[ANSWER] NO ... on reflection\n[ANSWER] YES", "YES"),  # last answer wins
    ("[ANSWER]\nYES/NO", None),                             # hedge
    ("I think the answer is yes.", None),
    ("", None),
    (None, None),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_prompts_never_leak(built):
    for name, (records, manifest) in built.items():
        task = manifest["domain"]["task"]
        for r in records[::7]:
            for with_expl in (True, False):
                for instruction in ("follow", "neutral"):
                    prompt = build_prompt(r, task, instruction=instruction, with_explanation=with_expl)
                    assert r["description_cx"] in prompt
                    assert (r["explanation_text"] in prompt) == with_expl


def test_leak_check_catches_true_rule():
    record = {"record_id": "x", "rule_formal": "credit_score >= 620", "described_formal": "income_k > 50"}
    with pytest.raises(AssertionError):
        check_no_leak(record, "... credit_score >= 620 ...")
    check_no_leak(record, "... income_k > 50 ...")
