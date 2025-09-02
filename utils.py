import os
import sys
import time

CUSTOM_TEMPLATES = {
    'oxford_pets': 'a photo of a {}, a type of pet.',
    'oxford_flowers': 'a photo of a {}, a type of flower.',
    'fgvc_aircraft': 'a photo of a {}, a type of aircraft.',
    'dtd': '{} texture.',
    'eurosat': 'a centered satellite photo of {}.',
    'stanford_cars': 'a photo of a {}.',
    'food101': 'a photo of {}, a type of food.',
    'sum397': 'a photo of a {}.',
    'caltech101': 'a photo of a {}.',
    'ucf101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = os.path.join(output, "log.txt")

    if os.path.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath)