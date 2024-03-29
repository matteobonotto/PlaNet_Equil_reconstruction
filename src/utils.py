from time import time
from pathlib import Path
# from lightning.pytorch.cli import LightningCLI 


def timer_func(func):
    """
    Decorator to time function execution

    :param func: function to be timed
    :return: timed function

    """
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'{func.__name__}() executed in {(t2-t1):.6f}s')
        print('')
        return result
    return wrapper

def touch_dir(dir_path: str) -> None:
    """
    Create dir if not exist

    :param dir_path: directory path

    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)



# class CustomCli(LightningCLI):
#     def add_arguments_to_parser(self, parser) -> None:
#         pass
