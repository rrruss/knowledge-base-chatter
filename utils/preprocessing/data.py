import torch


def convert_str_indices_to_token_indices(fast_tokenizer,
                                         question_context,
                                         start_end_str_indices,
                                         test=False,
                                         **tokenizer_kwargs):
    """
    Converts string indices common in QA tasks to token indices.

    Parameters
    ----------
    fast_tokenizer : instance of PreTrainedTokenizerFast
        We have to use a fast tokenizer in order to access offset mappings.
    question_context : str or list of (single) list of 2 strings (for
        2 sent tasks)
        The full text of the question and context or just the context,
        depending on the situation.
    start_end_str_indices : sequence with two integers.
        Contains start string index and end string index of the answer.
    test : boolean
        When True, print the decoded answer (for sanity checking).
    tokenizer_kwargs : dict
        Any remaining keyword arguments.

    Example
    -------
    >>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    >>> question_context = [['How many toys are on the floor?',
                             'The floor was covered in toys. 15 of them to be exact.']]
    >>> converted = convert_str_indices_to_token_indices(tokenizer,
                                                         question_context,
                                                         [62, 64],
                                                         test=True)
    answer using token indices = 15
    (16, 18)

    Returns
    -------
    A tuple with the start token index and end token index.
    """
    tokenized = fast_tokenizer(question_context,
                               return_offsets_mapping=True,
                               return_tensors='pt',
                               **tokenizer_kwargs)
    if isinstance(question_context, list):
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        offset_mapping = tokenized['offset_mapping']
        input_ids = tokenized['input_ids']
    else:
        offset_mapping = tokenized['offset_mapping'][0]
        input_ids = tokenized['input_ids'][0]

    span = [0, 0]
    offset_add, last_offset = 0, 0
    for i, offset in enumerate(offset_mapping):
        # print(start_end_str_indices, offset + offset_add, offset)
        if i > 0 and torch.equal(offset, torch.tensor([0, 0])):
            offset_add += last_offset
        elif torch.equal(offset, torch.tensor([0, 0])):
            continue
        if offset[0] + offset_add <= start_end_str_indices[0] <= offset[1] + offset_add:
            span[0] = i
        if offset[0] + offset_add <= start_end_str_indices[1] <= offset[1] + offset_add:
            span[1] = i
        if span[0] != 0 and span[1] != 0:
            break
        last_offset = offset[-1]
    if test:
        print('answer using token indices =',
              fast_tokenizer.decode(input_ids[span[0]:span[-1] + 1]))
    return span[0], span[-1]


if __name__ == '__main__':
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    question_context = [['How many toys are on the floor?',
                         'The floor was covered in toys. 15 of them to be exact.']]
    converted = convert_str_indices_to_token_indices(tokenizer,
                                                     question_context,
                                                     [62, 64],
                                                     test=True)
    print(converted)

    question_context = 'How many toys are on the floor? The floor was covered in toys. 15 of them to be exact.'
    converted = convert_str_indices_to_token_indices(tokenizer,
                                                     question_context,
                                                     [63, 72],
                                                     test=True)
    print(converted)
