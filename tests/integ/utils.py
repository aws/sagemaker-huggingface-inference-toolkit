from __future__ import absolute_import

import logging
import re
import signal
from contextlib import contextmanager
from time import time

from botocore.exceptions import ClientError


LOGGER = logging.getLogger("timeout")


class TimeoutError(Exception):
    pass


def clean_up(endpoint_name, sagemaker_session):
    try:
        sagemaker_session.delete_endpoint(endpoint_name)
        sagemaker_session.delete_endpoint_config(endpoint_name)
        sagemaker_session.delete_model(endpoint_name)
        LOGGER.info("deleted endpoint {}".format(endpoint_name))
    except ClientError as ce:
        if ce.response["Error"]["Code"] == "ValidationException":
            # avoids the inner exception to be overwritten
            pass


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):
        raise TimeoutError("timed out after {} seconds".format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(limit)

        yield
    finally:
        signal.alarm(0)


@contextmanager
def timeout_and_delete_by_name(endpoint_name, sagemaker_session, seconds=0, minutes=0, hours=0):
    with timeout(seconds=seconds, minutes=minutes, hours=hours) as t:
        try:
            yield [t]
        finally:
            clean_up(endpoint_name, sagemaker_session)


@contextmanager
def track_infer_time(buffer=[]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


_re_word_boundaries = re.compile(r"\b")


def count_tokens(inputs: dict, task: str) -> int:
    if task == "question-answering":
        context_len = len(_re_word_boundaries.findall(inputs["context"])) >> 1
        question_len = len(_re_word_boundaries.findall(inputs["question"])) >> 1
        return question_len + context_len
    else:
        return len(_re_word_boundaries.findall(inputs)) >> 1


def validate_text_classification(result=None, snapshot=None):
    for idx, _ in enumerate(result):
        assert result[idx].keys() == snapshot[idx].keys()
        assert result[idx]["score"] >= snapshot[idx]["score"]
    return True


def validate_zero_shot_classification(result=None, snapshot=None):
    assert result.keys() == snapshot.keys()
    assert result["labels"] == snapshot["labels"]
    assert result["sequence"] == snapshot["sequence"]
    for idx in range(len(result["scores"])):
        assert result["scores"][idx] >= snapshot["scores"][idx]
    return True


def validate_ner(result=None, snapshot=None):
    for idx, _ in enumerate(result):
        assert result[idx].keys() == snapshot[idx].keys()
        assert result[idx]["score"] >= snapshot[idx]["score"]
        assert result[idx]["entity"] == snapshot[idx]["entity"]
        assert result[idx]["entity"] == snapshot[idx]["entity"]
    return True


def validate_question_answering(result=None, snapshot=None):
    assert result.keys() == snapshot.keys()
    assert result["answer"] == snapshot["answer"]
    assert result["score"] >= snapshot["score"]
    return True


def validate_summarization(result=None, snapshot=None):
    assert result == snapshot
    return True


def validate_text2text_generation(result=None, snapshot=None):
    assert result == snapshot
    return True


def validate_translation(result=None, snapshot=None):
    assert result == snapshot
    return True


def validate_text_generation(result=None, snapshot=None):
    assert result is not None
    return True


def validate_feature_extraction(result=None, snapshot=None):
    assert result is not None
    return True


def validate_fill_mask(result=None, snapshot=None):
    assert result is not None
    return True
