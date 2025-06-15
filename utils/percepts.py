class Percepts:
    """Validate a VLM answer against the allowed objects."""
    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [o.strip().lower() for o in objects_of_interest]

    def validate_percept(self, model_answer: str) -> str | None:
        ans = model_answer.strip().lower()
        for obj in self._objects:
            if ans in obj or obj in ans:
                return obj
        return None