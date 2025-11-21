# ai_tutor/models/__init__.py

"""
Model loading utilities for the AI Tutor project.

Includes:
- Base model loader
- LoRA adapter loader
- Unified inference interface
"""

from .base_loader import load_base_model
from .lora_loader import load_finetuned_model
from .inference import generate_answer
