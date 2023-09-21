def model_fn(model_dir, context = None):
    return "model"


def input_fn(data, content_type, context = None):
    return "data"


def predict_fn(data, model, context = None):
    return "output"


def output_fn(prediction, accept, context = None):
    return prediction
