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

            ALWAYS START YOUR ANSWER EXACTLY LIKE THIS:
            "1. Core Idea"

            Then follow this structure:

            1. Core Idea
               - 2–4 short sentences, no jargon.
            2. Step-by-Step Example
               - A small code snippet.
               - Explain the code line by line.
            3. Common Mistake + Check-Your-Understanding Question
               - Mention one common beginner mistake.
               - End with a short question the student can answer.

            Use simple language, short paragraphs, and talk directly to the student ("you").
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

Tutor answer (follow the instructions above): [/INST]"""

    return prompt
