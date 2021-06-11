def load_fn(model_dir):
    return "model"


def preprocess_fn(data, content_type):
    return "data"


def predict_fn(data):
    return "output"


def postprocess_fn(prediction, accept):
    return prediction
