def model_fn(self, model_dir):
    return self.context.model_name


def input_fn(data, content_type):
    return "data"


def predict_fn(data, model):
    return "output"


def output_fn(prediction, accept):
    return prediction
