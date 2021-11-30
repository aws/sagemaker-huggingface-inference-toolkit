import os


def model_fn(self, model_dir):
    return f"Loading {os.path.basename(__file__)}"


def transform_fn(a, b, c, d):
    return f"output {b}"
