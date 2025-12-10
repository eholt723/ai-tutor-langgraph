# ai_tutor/prompts.py
from textwrap import dedent
from typing import Literal, Optional

Mode = Literal["base", "finetuned"]


def build_prompt(
    question: str,
    mode: Mode,
    context: Optional[str] = None,
) -> str:
    """
    Build a TinyLlama-style chat prompt.
    Finetuned mode has a much stronger tutoring style.
    """

    if mode == "finetuned":
        system = dedent(
            """
            You are a friendly, beginner-focused programming tutor.
            Your students are taking an introductory course in Python and basic OOP.

            Always answer in EXACTLY three numbered sections with these headings:

            1. Core Idea
            2. Step-by-Step Example
            3. Common Mistake + Check-Your-Understanding Question

            Requirements:

            - Do NOT include any extra headings, preamble, or closing summary.
            - Do NOT repeat the phrase "Tutor answer" or the instructions.
            - Start your answer directly with: 1. Core Idea
            - Keep language simple and avoid jargon.
            - In the Step-by-Step Example, include a small code snippet and explain it line by line.
            - In the Common Mistake section, mention one typical beginner error and end with a short question the student can answer.

            Talk directly to the student ("you") and keep paragraphs short.
            """
        )
    else:
        system = dedent(
            """
            You are a helpful programming assistant.
            Answer clearly and concisely.
            """
        )

    ctx_block = f"\n\nContext for the tutor (reference notes):\n{context}" if context else ""

    prompt = f"""<s>[INST] <<SYS>>
{system}
<<SYS>>

Student question:
{question}{ctx_block}

Write your answer now. [/INST]"""

    return prompt
