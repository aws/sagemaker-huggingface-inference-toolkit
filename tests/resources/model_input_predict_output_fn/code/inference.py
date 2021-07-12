def model_fn(model_dir):
    return "model"


def input_fn(data, content_type):
    return "data"


def predict_fn(data, model):
    return "output"


def output_fn(prediction, accept):
    return prediction
