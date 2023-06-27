import importlib.util
import os

_optimum_neuron = importlib.util.find_spec("optimum.neuron") is not None


def is_optimum_neuron_available():
    return _optimum_neuron


OPTIMUM_NEURON_TASKS = [
    "feature-extraction",
    "fill-mask",
    "text-classification",
    "token-classification",
    "question-answering",
    "zero-shot-classification",
]


def get_input_shapes(model_dir):
    """Method to get input shapes from model config file. If config file is not present, default values are returned."""
    from transformers import AutoConfig

    input_shapes = {}
    input_shapes_available = False
    # try to get input shapes from config file
    try:
        config = AutoConfig.from_pretrained(model_dir)
        if hasattr(config, "neuron_batch_size") and hasattr(config, "neuron_sequence_length"):
            input_shapes["batch_size"] = config.neuron_batch_size
            input_shapes["sequence_length"] = config.neuron_sequence_length
            input_shapes_available = True
    except:
        input_shapes_available = False

    # return input shapes if available
    if input_shapes_available:
        return input_shapes

    # extract input shapes from environment variables
    sequence_length = os.environ.get("OPTIMUM_NEURON_SEQUENCE_LENGTH", None)
    if not int(sequence_length) > 0:
        raise ValueError(
            f"OPTIMUM_NEURON_SEQUENCE_LENGTH must be set to a positive integer. Current value is {sequence_length}"
        )
    batch_size = os.environ.get("OPTIMUM_NEURON_BATCH_SIZE", 1)

    return {"batch_size": int(batch_size), "sequence_length": int(sequence_length)}


def get_optimum_neuron_pipeline(task, model_dir):
    """Method to get optimum neuron pipeline for a given task. Method checks if task is supported by optimum neuron and if required environment variables are set, in case model is not converted. If all checks pass, optimum neuron pipeline is returned. If checks fail, an error is raised."""
    from optimum.neuron.pipelines import pipeline
    from optimum.neuron.utils import NEURON_FILE_NAME

    # check task support
    if task not in OPTIMUM_NEURON_TASKS:
        raise ValueError(
            f"Task {task} is not supported by optimum neuron and inf2. Supported tasks are: {OPTIMUM_NEURON_TASKS}"
        )

    # check if model is already converted and has support input shapes available
    export = True
    if NEURON_FILE_NAME in os.listdir(model_dir):
        export = False

    # get static input shapes to run inference
    input_shapes = get_input_shapes(model_dir)
    # get optimum neuron pipeline
    neuron_pipe = pipeline(task, model=model_dir, export=export, input_shapes=input_shapes)

    return neuron_pipe
