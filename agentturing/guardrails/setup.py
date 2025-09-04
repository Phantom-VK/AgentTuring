import re
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, DetectPII, GibberishText

def math_intent_check(text: str) -> bool:
    """Lightweight math-only heuristic."""
    math_keywords = [
        "solve","calculate","find","derive","integrate","differentiate",
        "equation","inequality","factor","simplify","proof","theorem",
        "matrix","vector","probability","expectation","variance","limit",
        "derivative","integral","gradient","hessian","algebra","geometry",
        "trigonometry","calculus","number theory","combinatorics", "cube", "square", "subtraction",

    ]
    t = text.lower()
    has_kw = any(k in t for k in math_keywords)
    has_sym = bool(re.search(r"[+\-*/=^(){}\[\]√∑∫]|\bpi\b|\btheta\b|\d", t))
    return has_kw or has_sym

def _to_string(validated):
    """Extract string from Guardrails ValidationOutcome or passthrough str."""
    # Guard.validate may return a ValidationOutcome with `.validated_output`
    if hasattr(validated, "validated_output"):
        return validated.validated_output
    return validated if isinstance(validated, str) else str(validated)

def make_input_guard():
    """Create an input guard and return a callable that outputs a string."""
    guard = (
        Guard()
        .use(ToxicLanguage(threshold=0.7, validation_method="sentence", on_fail=OnFailAction.EXCEPTION))
        .use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.FILTER))
    )

    def validate_input(text: str) -> str:
        if not math_intent_check(text):
            raise ValueError("Only math-related questions are allowed.")
        result = guard.validate(text)
        return _to_string(result)

    return validate_input

def make_output_guard():
    """Create an output guard and return a callable that outputs a string."""
    guard = (
        Guard()
        .use(ToxicLanguage(threshold=0.7, validation_method="sentence", on_fail=OnFailAction.FILTER))
        .use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.FILTER))
    )

    def validate_output(text: str) -> str:
        result = guard.validate(text)
        return _to_string(result)

    return validate_output
