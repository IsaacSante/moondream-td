class Prompt:
    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [obj.strip() for obj in objects_of_interest]

    @property
    def text(self) -> str:
        objects_part = " or ".join(self._objects)
        return f"Is a person grabbing in the palm of their hand? Say yes or no."
