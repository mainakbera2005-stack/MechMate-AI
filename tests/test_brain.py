from brain import _prepare_request_payload, ENGINEERING_RULES


def test_engineering_rules_content():
    assert "NUMERICAL VERIFICATION & CALCULATION DISCIPLINE" in ENGINEERING_RULES
    assert "KaTeX-compatible LaTeX" in ENGINEERING_RULES
    assert "$$ ... $$" in ENGINEERING_RULES


def test_prepare_request_payload():
    combined_prompt, prompt = _prepare_request_payload(
        question="Calculate the Reynolds number for water flow.",
        context="Pipe diameter is 0.05m, velocity is 2m/s.",
        history="",
        system_prompt="You are MechMate AI."
    )

    assert "Reynolds number" in prompt
    assert "Pipe diameter" in prompt
    assert "NUMERICAL VERIFICATION" in combined_prompt
