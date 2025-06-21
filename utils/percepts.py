class Percepts:
    """Validate a VLM answer against the allowed objects."""

    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [o.strip().lower() for o in objects_of_interest]

    def validate_percept(self, model_answer: str) -> str | None:
        ans = model_answer.strip().lower()

        # Check if it's "none" or similar
        if "none" in ans or "no" in ans or "not holding" in ans:
            return None

        # Check for each object
        for obj in self._objects:
            if obj in ans:
                # Try to extract confidence if present
                import re
                confidence_match = re.search(r'(\d+)%?', ans)
                if confidence_match:
                    confidence = int(confidence_match.group(1))
                    return {"object": obj, "confidence": confidence}
                else:
                    return {"object": obj, "confidence": None}

        return None
