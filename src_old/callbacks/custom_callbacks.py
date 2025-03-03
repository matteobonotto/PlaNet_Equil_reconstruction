from lightning.pytorch.callbacks import TQDMProgressBar


class CustomCallback(TQDMProgressBar):
    """
    Example of custom callback
    """

    def __init__(self, refresh_rate):
        super(CustomCallback, self).__init__(refresh_rate=refresh_rate)
