# called after td has saved the current image to disk and you can now call on moondream process to do detection.
class TDEvent():
    pass

# called after moondream completes obj detection to start upstream process back into td.
class MoondreamEvent():
    pass


