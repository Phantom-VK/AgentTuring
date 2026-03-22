"""Lightweight input and output guard helpers for math-only requests."""

import re


def math_intent_check(text: str) -> bool:
    """Lightweight math-only heuristic."""
    math_keywords = [
        "solve", "calculate", "find", "derive", "integrate", "differentiate",
        "equation", "inequality", "factor", "simplify", "proof", "theorem",
        "matrix", "vector", "probability", "expectation", "variance", "limit",
        "derivative", "integral", "gradient", "hessian", "algebra", "geometry",
        "trigonometry", "calculus", "number theory", "combinatorics",
        "cube", "square", "subtraction",
        "theory", "concept", "arrange", "math", "ways", "formula", "quadratic",
    ]
    lowered = text.lower()
    has_keyword = any(keyword in lowered for keyword in math_keywords)
    has_symbol = bool(re.search(r"[+\-*/=^(){}\[\]√∑∫]|\bpi\b|\btheta\b|\d", lowered))
    return has_keyword or has_symbol


def _contains_toxicity(text: str) -> bool:
    toxic_markers = (
        "kill",
        "hate",
        "terrorist",
        "bomb",
        "self-harm",
        "suicide",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in toxic_markers)


def _filter_pii(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email redacted]", text)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)", "[phone redacted]", text)
    return text


def make_input_guard():
    """Create an input guard and return a callable that outputs a string."""

    def validate_input(text: str) -> str:
        if not math_intent_check(text):
            raise ValueError("Only math-related questions are allowed.")

        sanitized = _filter_pii(text.strip())
        if _contains_toxicity(sanitized):
            raise ValueError("Unsafe language detected in request.")

        return sanitized

    return validate_input


def make_output_guard():
    """Create an output guard and return a callable that outputs a string."""

    def validate_output(text: str) -> str:
        sanitized = _filter_pii(text.strip())
        if _contains_toxicity(sanitized):
            return (
                "The generated answer did not meet safety requirements. "
                "Please rephrase the question."
            )
        return sanitized

    return validate_output
