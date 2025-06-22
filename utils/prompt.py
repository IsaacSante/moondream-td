class Prompt:
    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [obj.strip() for obj in objects_of_interest]

    @property
    def text(self) -> str:
        objects_part = " or ".join(self._objects)
        return f"Is the person holding something? If not, respond with 'none'. If they hold any of the objects, a {objects_part}, respond with the object name."
