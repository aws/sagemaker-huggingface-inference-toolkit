from integ.utils import (
    validate_feature_extraction,
    validate_fill_mask,
    validate_ner,
    validate_question_answering,
    validate_summarization,
    validate_text2text_generation,
    validate_text_classification,
    validate_text_generation,
    validate_translation,
    validate_zero_shot_classification,
)


task2model = {
    "text-classification": {
        "pytorch": "distilbert-base-uncased-finetuned-sst-2-english",
        "tensorflow": "distilbert-base-uncased-finetuned-sst-2-english",
    },
    "zero-shot-classification": {
        "pytorch": "joeddav/xlm-roberta-large-xnli",
        "tensorflow": None,
    },
    "feature-extraction": {
        "pytorch": "bert-base-uncased",
        "tensorflow": "bert-base-uncased",
    },
    "ner": {
        "pytorch": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "tensorflow": "dbmdz/bert-large-cased-finetuned-conll03-english",
    },
    "question-answering": {
        "pytorch": "distilbert-base-uncased-distilled-squad",
        "tensorflow": "distilbert-base-uncased-distilled-squad",
    },
    "fill-mask": {
        "pytorch": "albert-base-v2",
        "tensorflow": "albert-base-v2",
    },
    "summarization": {
        "pytorch": "sshleifer/distilbart-xsum-1-1",
        "tensorflow": "sshleifer/distilbart-xsum-1-1",
    },
    "translation_xx_to_yy": {
        "pytorch": "Helsinki-NLP/opus-mt-en-de",
        "tensorflow": "Helsinki-NLP/opus-mt-en-de",
    },
    "text2text-generation": {
        "pytorch": "t5-small",
        "tensorflow": "t5-small",
    },
    "text-generation": {
        "pytorch": "gpt2",
        "tensorflow": "gpt2",
    },
}

task2input = {
    "text-classification": {"inputs": "I love you. I like you"},
    "zero-shot-classification": {
        "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
        "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
    },
    "feature-extraction": {"inputs": "What is the best book."},
    "ner": {"inputs": "My name is Wolfgang and I live in Berlin"},
    "question-answering": {
        "inputs": {
            "question": "What is used for inference?",
            "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference.",
        }
    },
    "fill-mask": {"inputs": "Paris is the [MASK] of France."},
    "summarization": {
        "inputs": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    },
    "translation_xx_to_yy": {"inputs": "My name is Sarah and I live in London"},
    "text2text-generation": {
        "inputs": "question: What is 42 context: 42 is the answer to life, the universe and everything."
    },
    "text-generation": {"inputs": "My name is philipp and I am"},
}

task2output = {
    "text-classification": [{"label": "POSITIVE", "score": 0.99}],
    "zero-shot-classification": {
        "sequence": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
        "labels": ["refund", "faq", "legal"],
        "scores": [0.96, 0.027, 0.008],
    },
    "ner": [
        {"word": "Wolfgang", "score": 0.99, "entity": "I-PER", "index": 4, "start": 11, "end": 19},
        {"word": "Berlin", "score": 0.99, "entity": "I-LOC", "index": 9, "start": 34, "end": 40},
    ],
    "question-answering": {"score": 0.99, "start": 68, "end": 77, "answer": "sagemaker"},
    "summarization": [{"summary_text": " The A The The ANew York City has been installed in the US."}],
    "translation_xx_to_yy": [{"translation_text": "Mein Name ist Sarah und ich lebe in London"}],
    "text2text-generation": [{"generated_text": "42 is the answer to life, the universe and everything"}],
    "feature-extraction": None,
    "fill-mask": None,
    "text-generation": None,
}

task2performance = {
    "text-classification": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 1,
        },
    },
    "zero-shot-classification": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 4,
        },
    },
    "feature-extraction": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 1,
        },
    },
    "ner": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 1,
        },
    },
    "question-answering": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 4,
        },
    },
    "fill-mask": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 3,
        },
    },
    "summarization": {
        "cpu": {
            "average_request_time": 26,
        },
        "gpu": {
            "average_request_time": 3,
        },
    },
    "translation_xx_to_yy": {
        "cpu": {
            "average_request_time": 8,
        },
        "gpu": {
            "average_request_time": 3,
        },
    },
    "text2text-generation": {
        "cpu": {
            "average_request_time": 4,
        },
        "gpu": {
            "average_request_time": 3,
        },
    },
    "text-generation": {
        "cpu": {
            "average_request_time": 15,
        },
        "gpu": {
            "average_request_time": 3,
        },
    },
}

task2validation = {
    "text-classification": validate_text_classification,
    "zero-shot-classification": validate_zero_shot_classification,
    "feature-extraction": validate_feature_extraction,
    "ner": validate_ner,
    "question-answering": validate_question_answering,
    "fill-mask": validate_fill_mask,
    "summarization": validate_summarization,
    "translation_xx_to_yy": validate_translation,
    "text2text-generation": validate_text2text_generation,
    "text-generation": validate_text_generation,
}
