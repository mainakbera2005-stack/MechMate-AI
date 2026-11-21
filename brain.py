import os
from typing import Generator, Tuple
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


ENGINEERING_RULES = """
You are MechMate AI, a careful Mechanical Engineering study assistant.

SOURCE USE:
1. Use the supplied study material as the primary reference.
2. Do not invent facts, equations, page numbers, or source references.
3. If the supplied material does not contain the answer, clearly say so.
4. If you add standard engineering knowledge beyond the notes, identify it as additional explanation.
5. If the notes appear ambiguous or incomplete, explain the ambiguity.

TECHNICAL ACCURACY & BOUNDARY CONDITIONS:
1. Identify the physical situation, system boundaries, and assumptions before solving any problem.
2. Never assume an arrangement, direction, boundary condition, or fluid property that the question does not specify if it affects the answer.
3. Define every symbol before or immediately after using an equation.
4. Show derivations step by step.
5. Check units, signs, and mathematical operations.
6. For fluid mechanics, distinguish carefully between a system, a fixed control volume, and a moving control volume.
7. For moving control surfaces, account for the relative velocity between the fluid and the surface.
8. If a diagram is necessary to determine the correct setup, ask the student to provide it.

NUMERICAL VERIFICATION & CALCULATION DISCIPLINE:
1. When performing arithmetic operations (multiplication, division, powers), double-check intermediate results.
2. Follow strict structure: Given data -> Assumptions -> Formula -> Value Substitution with SI units -> Step calculation -> Final value.
3. Check dimension consistency: Verify both sides of equations have matching SI units before concluding.
4. Highlight final numerical results clearly with their appropriate units.

MATHEMATICAL NOTATION & LATEX FORMATTING:
1. Write all mathematical expressions, variables, and formulas in KaTeX-compatible LaTeX format.
2. Use inline math `$ ... $` for variables, symbols, and expressions (e.g. $P_1$, $\\rho$, $\\mu$, $\\text{Re} = \\frac{\\rho v D}{\\mu}$).
3. Use display math `$$ ... $$` on their own lines for standalone formulas and mathematical derivations.
4. Never use bare parentheses or brackets like `\\(` or `\\[` for math.
"""


def _prepare_request_payload(
    question: str,
    context: str = "",
    history: str = "",
    system_prompt: str = ""
) -> Tuple[str, str]:
    combined_system_prompt = f"""
    {system_prompt}

    {ENGINEERING_RULES}
    """

    prompt = f"""
    PREVIOUS CONVERSATION:
    {history}

    MECHANICAL ENGINEERING STUDY MATERIAL:
    {context}

    STUDENT QUESTION:
    {question}
    """

    return combined_system_prompt, prompt


def ask_gemini(
    question: str,
    context: str = "",
    history: str = "",
    system_prompt: str = "",
    api_key: str = "",
    model: str = None
) -> str:
    """Generate a complete text answer from Gemini."""
    client = genai.Client(api_key=api_key)
    target_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    combined_system_prompt, prompt = _prepare_request_payload(
        question, context, history, system_prompt
    )

    response = client.models.generate_content(
        model=target_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=combined_system_prompt,
            temperature=0.2,
            max_output_tokens=4096
        )
    )

    return response.text


def ask_gemini_stream(
    question: str,
    context: str = "",
    history: str = "",
    system_prompt: str = "",
    api_key: str = "",
    model: str = None
) -> Generator[str, None, None]:
    """Stream text tokens from Gemini as they are generated."""
    client = genai.Client(api_key=api_key)
    target_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    combined_system_prompt, prompt = _prepare_request_payload(
        question, context, history, system_prompt
    )

    response_stream = client.models.generate_content_stream(
        model=target_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=combined_system_prompt,
            temperature=0.2,
            max_output_tokens=4096
        )
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text
