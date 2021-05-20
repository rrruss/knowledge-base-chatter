import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: move these global variables to local as we add metrics.
# TODO: finetune `MODEL` on the QA/dialog dataset for a better metric
MODEL = RobertaForMaskedLM.from_pretrained('roberta-large').to(DEVICE)
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')


def mlm_metric(prompt, response):
    """
    Masked language model metric.

    (Need to double-check implementation.)

    Parameters
    ----------
    prompt   - (str) The user comment.
    response - (str) The bot's response.

    Returns sum of negative probabilities (lower is better).
    -------

    """
    tokens = TOKENIZER([[prompt, response]], return_tensors='pt')
    input_ids = tokens['input_ids'][0]
    dividing_indices = [i for i in range(1, input_ids.shape[0]) if input_ids[i - 1:i + 1].tolist() == [2, 2]]
    response_start_index = dividing_indices[0] + 1
    probs = []
    for i, response_id in enumerate(input_ids[response_start_index:-1]):
        response_id = response_id.item()
        tokens['input_ids'][0][response_start_index + i] = TOKENIZER.mask_token_id
        with torch.no_grad():
            output = MODEL(**tokens.to(DEVICE))
            preds = torch.softmax(output.logits, dim=-1)[0][response_start_index + i]
            prob = preds[response_id].item()
        probs.append(prob)
    return -torch.sum(torch.tensor(probs))


if __name__ == '__main__':
    print(mlm_metric('How do I change my password?', 'You do not.'))
