class Prompt:
    _start = "If a person is holding one of these toys,"
    _stop = " return the toy name."
    _instruction = "Else return none."

    def __init__(self, objects_of_interest: list[str]) -> None:
        if not objects_of_interest:
            raise ValueError("objects_of_interest may not be empty")
        # make a copy and strip extraneous whitespace
        self._objects = [obj.strip() for obj in objects_of_interest]

    # public attribute for convenience
    @property
    def text(self) -> str:
        return self._construct_prompt()

    # internal helper
    def _construct_prompt(self) -> str:
        if len(self._objects) == 1:
            objects_part = self._objects[0]
        else:
            objects_part = ", ".join(self._objects[:-1]) + " and " + self._objects[-1]
        # guarantee single-spaced output
        return f"{self._start} {objects_part} {self._stop} {self._instruction}"
