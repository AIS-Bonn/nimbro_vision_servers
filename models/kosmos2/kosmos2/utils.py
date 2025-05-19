import re
from typing import Optional

PATTERNS = {
    "caption_brief":   re.compile(
        r"^\s*<grounding>\s*An image of(?:\s+.*)?\s*$", re.I),
    "caption_detailed": re.compile(
        r"^\s*<grounding>\s*Describe this image in detail:(?:\s+.*)?\s*$", re.I),
    "grounded_vqa":             re.compile(
        r"^\s*<grounding>\s*Question:\s*.+?\s*Answer:\s*$", re.I),
    "phrase_grounding": re.compile(
        r"^\s*<grounding>\s*<phrase>\s*.+?\s*</phrase>\s*$", re.I | re.S),
}

def get_prompt_type(text: str) -> Optional[str]:
    """
    Return the canonical name of the prompt format that `text`
    conforms to, or None if the string is not valid.

    >>> get_prompt_type("<grounding> An image of a cat")
    'caption_brief'
    >>> get_prompt_type("<grounding> Describe this image in detail:")
    'caption_detailed'
    >>> get_prompt_type("<grounding> Question: What is special? Answer:")
    'grounded_vqa'
    >>> get_prompt_type("<grounding><phrase> a snowman </phrase>")
    'phrase_grounding'
    """
    for name, pattern in PATTERNS.items():
        if pattern.fullmatch(text):
            return name
    return None