import re

STEP_HEADER_PAT = re.compile(r"(?:^|\n)(?:Answer step-by-step:|Steps?:)", re.IGNORECASE)
FINAL_ANSWER_PAT = re.compile(r"(?:^|\n)(?:Final Answer:|Answer:)", re.IGNORECASE)

def extract_steps(text: str) -> str:
    """Return only the step-by-step solution section."""
    if not isinstance(text, str):
        text = str(text)

    # 1) Cut off everything before the steps header if present
    m = STEP_HEADER_PAT.search(text)
    if m:
        text = text[m.end():].strip()

    # 2) Stop at first explicit final answer or next question block
    stop_idx = None
    for pat in [FINAL_ANSWER_PAT, re.compile(r"(?:^|\n)Question:", re.IGNORECASE)]:
        mm = pat.search(text)
        if mm:
            stop_idx = mm.start()
            break
    if stop_idx is not None:
        text = text[:stop_idx].strip()

    # 3) Remove leading System/Human echoes and excess whitespace
    text = re.sub(r"(?s)^System:.*?(?:\n|$)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"(?s)^Human:.*?(?:\n|$)", "", text, flags=re.IGNORECASE).strip()

    # 4) Normalize numbered bullets
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Deduplicate consecutive identical lines
    cleaned = []
    for ln in lines:
        if not cleaned or cleaned[-1] != ln:
            cleaned.append(ln)

    # Ensure it begins with a steps header
    if cleaned and not cleaned[0].lower().startswith(("step", "answer step-by-step")):
        cleaned.insert(0, "Steps:")

    return "\n".join(cleaned)
