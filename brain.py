
import os

from google import genai
from google.genai import types


# ---------------------------------------------------------
# 1. GEMINI MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


# ---------------------------------------------------------
# 2. MECHANICAL ENGINEERING INSTRUCTIONS
# ---------------------------------------------------------

ENGINEERING_RULES = """
You are MechMate AI, a careful Mechanical Engineering
study assistant for undergraduate students.

SOURCE AND CITATION RULES:

1. Use supplied study material as the primary reference.
2. Retrieved sources are identified as SOURCE 1,
   SOURCE 2, SOURCE 3, etc.
3. Cite relevant statements using this exact format:
   [Source 1, p. 6]
4. Use only source numbers and page numbers supplied
   in the retrieved context.
5. Never invent citations, page numbers or references.
6. If uploaded material does not contain the answer,
   clearly disclose this.
7. Clearly distinguish uploaded study material from
   additional general engineering knowledge.
8. Do not claim that a statement comes from a PDF
   unless the retrieved content supports it.

TECHNICAL ACCURACY:

1. Identify the physical situation and assumptions
   before solving a numerical problem.
2. Never assume an arrangement, direction, boundary
   condition or fluid property when it affects
   the answer and is not specified.
3. Define every symbol used in an equation.
4. Show derivations and calculations step by step.
5. Check units, signs and mathematical operations.
6. For fluid mechanics, distinguish between a system,
   a fixed control volume and a moving control volume.
7. For moving control surfaces, account for relative
   velocity between the fluid and the surface.
8. If a diagram is necessary, ask the student to
   provide it rather than guessing.

ANSWER STYLE:

1. Explain concepts in beginner-friendly language.
2. Use clear headings and structured explanations.
3. For numerical problems, use:
   - Given
   - Required
   - Assumptions
   - Formula
   - Substitution
   - Step-by-step solution
   - Final answer with SI units
4. Explain equations and their physical significance.
5. Include practical engineering applications.
6. Highlight the final answer.
7. Prefer a correct, qualified answer over an
   unsupported confident answer.
8. Do not claim an answer is verified unless
   it has actually been checked.

IMPORTANT:

Your mission is to make Mechanical Engineering
easy, understandable and interesting.

Act as a teacher, mentor and study companion.
"""


# ---------------------------------------------------------
# 3. GEMINI RESPONSE GENERATION
# ---------------------------------------------------------

def ask_gemini(
    question,
    context="",
    history="",
    system_prompt="",
    api_key="",
):
    """
    Generate a Mechanical Engineering answer using Gemini.
    """

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    # Initialize Gemini client.
    client = genai.Client(api_key=api_key)

    # Combine application-level and engineering instructions.
    combined_system_prompt = "\n\n".join(
        part.strip()
        for part in (system_prompt, ENGINEERING_RULES)
        if part and part.strip()
    )

    # Construct the student prompt.
    prompt = f"""
PREVIOUS CONVERSATION:

{history or "No previous conversation."}


RETRIEVED MECHANICAL ENGINEERING STUDY MATERIAL:

{context or "No relevant uploaded study material was retrieved."}


STUDENT QUESTION:

{question.strip()}


RESPONSE REQUIREMENTS:

- Answer the student's actual question.
- Prioritize relevant uploaded study material.
- Include accurate inline citations wherever supported.
- Clearly label additional general engineering knowledge.
- Never invent source references or page numbers.
- Explain technical concepts in an understandable way.
""".strip()

    # Generate the answer.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=combined_system_prompt,
            temperature=0.2,
            max_output_tokens=1800,
        ),
    )

    # Extract the generated answer.
    answer = getattr(response, "text", None)

    if not answer or not answer.strip():
        raise RuntimeError(
            "Gemini returned an empty response. "
            "Check model availability, API quota and configuration."
        )

    return answer.strip()
