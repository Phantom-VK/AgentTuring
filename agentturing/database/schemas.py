from typing_extensions import TypedDict, List


class State(TypedDict):
    question: str
    context: List[str]
    answer: str
    next_step: str
