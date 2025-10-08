import re


def normalize_text_ascii_letters_only(text: str) -> str:
    """
    Normalize text to mirror the training pipeline:
    - lowercase
    - keep ASCII letters and spaces only (drop punctuation/digits/emojis)
    - collapse whitespace
    """
    x = text.lower()
    x = re.sub(r"[^a-z\s]", " ", x)          # letters + spaces
    x = re.sub(r"\s+", " ", x).strip()
    return x
