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
    Build prompts for TinyLlama.

    - finetuned: strict 1/2/3 tutoring structure with a chat-style template.
    - base: simple Q&A completion prompt.
    """

    if mode == "finetuned":
        # STRONG tutoring instructions for the LoRA model
        system = dedent(
            """
            You are a friendly, beginner-focused programming tutor.
            Your students are taking an introductory course in Python and basic OOP.

            Your answers MUST ALWAYS use EXACTLY three numbered sections
            with these headings, in this order:

            1. Core Idea
            2. Step-by-Step Example
            3. Common Mistake + Check-Your-Understanding Question

            Formatting rules:

            - Do NOT write anything before "1. Core Idea".
            - Do NOT add extra headings or a closing summary.
            - Do NOT include labels like "Student answer:" or "Tutor answer:".
            - Use short paragraphs and plain language.

            Content rules:

            1. Core Idea
               - Explain the main concept in 2–4 short sentences.
               - Assume the student is a beginner.

            2. Step-by-Step Example
               - Show a small Python code snippet in a fenced block:
                 ```python
                 # code here
                 ```
               - Explain what happens line by line in simple terms.

            3. Common Mistake + Check-Your-Understanding Question
               - Describe one typical beginner mistake related to this concept.
               - End with a short question the student can answer to check understanding.

            Always talk directly to the student using "you".
            """
        )

        ctx_block = (
            f"\n\nContext for the tutor (reference notes):\n{context}"
            if context
            else ""
        )

        prompt = f"""<s>[INST] <<SYS>>
{system}
<<SYS>>

Student question:
{question}{ctx_block}

Write your answer now. [/INST]"""
        return prompt

    # -------- BASE MODEL PROMPT (simple completion) --------
    system = dedent(
        """
        You are a helpful programming assistant.
        Answer clearly and concisely in a way a beginner can understand.
        """
    )

    ctx_block = (
        f"\n\nExtra reference notes (optional, may be empty):\n{context}"
        if context
        else ""
    )

    # Simple, non-chat prompt for better base completions
    prompt = f"""{system}

Question:
{question}{ctx_block}

Answer:"""

    return prompt
