import os


def model_fn(model_dir, context = None):
    return f"Loading {os.path.basename(__file__)}"


def transform_fn(a, b, c, d, context = None):
    return f"output {b}"
