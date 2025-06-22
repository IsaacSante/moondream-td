class Prompt:
    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        self._objects = [obj.strip() for obj in objects_of_interest]

    @property
    def text(self) -> str:
        objects_part = " or ".join(self._objects)
        return f"Is a person grabbing of these objects {objects_part}? If they are then say which object. If they are not then say 'none.' Important: Only say the object if the persons hand is grabbing the object. It must be grabbing the object. If it in frame and not grabbing the object then say 'none.' "
