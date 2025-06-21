class Prompt:
    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [obj.strip() for obj in objects_of_interest]

    @property
    def text(self) -> str:
        objects_part = " or ".join(self._objects)
        return f"Is the person holding {objects_part}? If yes, respond with the object name and confidence percentage (0-100). If no, respond with 'none'."
