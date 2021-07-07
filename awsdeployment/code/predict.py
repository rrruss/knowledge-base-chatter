import os
import re

import json
import logging
import torch
import torch.nn as nn


# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    logging.info("Loading model.")
    logging.info("model_dir = " + model_dir)
    logging.info("model_dir contents = \n" + '\n'.join(os.listdir(model_dir)))

    try:
        model = Model()
        logging.info("Model initialized.")
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        logging.info("State dict loaded.")
    except:
        logging.exception('message')
    logging.info("Torch loaded model.")
    model.eval()
    logging.info(model.context_embeddings)

    logging.info("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logging.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        output = json.dumps(prediction)
        return output, accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))


def predict_fn(input_dict, model):
    print('Inferring class of input data.')

    start_logits, end_logits, input_ids, relevance_logits = model(input_dict['question'])
    topk_relevant = torch.topk(relevance_logits, k=relevance_logits.shape[0] // 2, dim=-1)
    starts = torch.argmax(start_logits, dim=-1)
    ends = torch.argmax(end_logits, dim=-1)
    best_answers = []
    for index in topk_relevant.indices:
        question_context = input_ids[index]
        start = starts[index]
        end = ends[index]
        best_answers.append(model.r_tokenizer.decode(question_context[start:end + 1]))

    return best_answers
