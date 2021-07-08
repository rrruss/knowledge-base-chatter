import json
import os
import re

import joblib
import torch
import torch.nn as nn
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                          DPRReader, DPRReaderTokenizerFast)
from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast, RobertaForSequenceClassification


class LongQAModel(nn.Module):
    """
    A full dense passage retrieval model.

    Parameters
    ----------
    contexts : sequence of strings
        The (exhaustive) contexts that comprise the full documentation.
    fill_context_embeddings : boolean
        Whether to fill the `context_embeddings` (which could take a
        long time) or not. This should be set to False in the `model_fn`
        in the predict.py script for SageMaker, since SageMaker will
        otherwise time out.
    """

    def __init__(self,
                 contexts=None,
                 fill_context_embeddings=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(LongQAModel, self).__init__()
        self.device = device
        self.c_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
        self.c_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.q_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.r_model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to(device)
        self.r_tokenizer = DPRReaderTokenizerFast.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.contexts = contexts
        # Not enough time to load context embeddings in AWS SageMaker,
        # but can fill weights from saved state dict after loading model.
        if not self.contexts:
            with open('code/contexts.json') as f:
                self.contexts = json.load(f)
#             output_features = self.c_model.ctx_encoder.bert_model.pooler.dense.out_features
#             self.context_embeddings = nn.Parameter(torch.zeros(len(self.contexts), output_features)).to(device)
#         else:
        context_embeddings = []
        with torch.no_grad():
           for context in self.contexts:
               input_ids = self.c_tokenizer(context, return_tensors='pt').to(device)["input_ids"]
               output = self.c_model(input_ids)
               context_embeddings.append(output.pooler_output)
        self.context_embeddings = nn.Parameter(torch.cat(context_embeddings, dim=0)).to(device)
        print('cwd!:', os.getcwd())
        print(os.listdir('code'))
        self.noise_remover = joblib.load('code/filter_model.sav')

    def forward(self, question, retrieval_only=False):
        question_sentences = re.findall(r'[^.!?\n]+[.!?]', question)
        filtered_sentences = []
        for sentence in question_sentences:
            if (self.noise_remover.predict([sentence])) == 'Relevant':
                filtered_sentences.append(sentence)
        question = ' '.join(filtered_sentences) or question  # if filtered sents removes everything, use question
        q_input_ids = self.q_tokenizer(question, return_tensors='pt').to(self.device)['input_ids']
        q_output = self.q_model(q_input_ids)
        q_embedding = q_output.pooler_output
        similarities = torch.matmul(q_embedding, self.context_embeddings.T)
        topk_similarities = torch.topk(similarities[0], k=10, dim=-1)
        contexts = [self.contexts[i] for i in topk_similarities.indices]
        if retrieval_only:
            return contexts

        encoded_inputs = self.r_tokenizer(
            questions=[question for _ in contexts],
            # titles=[],  # add if contexts have titles
            texts=contexts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        r_output = self.r_model(**encoded_inputs)
        return r_output.start_logits, r_output.end_logits, encoded_inputs['input_ids'], r_output.relevance_logits


class BaselineQAModel(nn.Module):
    """
    The baseline model for the question-answering task.

    Parameters
    ----------
    device : torch.device
        The device to run the model and inferences on.
    model_class : Hugging Face model with a span classification head on top
        A model such as BertForQuestionAnswering.
    max_length : The maximum number of tokens to use in one pass. (All tokens
        will be covered, since overflow tokens are also passed through the
        model.)
    model_tokenizer : Hugging Face tokenizer
        Tested with non-Rust based tokenizers at present.
    pretrained_model_path : str
        Path to a pretrained model (either in the cloud or on one's machine).
    stride : int
        The amount of overlap between overflow tokens and the tokens already
        passed into the model.
    """

    def __init__(self,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model_class=RobertaForQuestionAnswering,
                 max_length=None,
                 model_tokenizer=RobertaTokenizerFast,
                 pretrained_model_path='roberta-large',
                 stride=12):
        super(BaselineQAModel, self).__init__()
        self.device = device
        self.tokenizer = model_tokenizer.from_pretrained(pretrained_model_path)
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length // 2
        self.stride = stride
        self.model = model_class.from_pretrained(pretrained_model_path).to(self.device)

    def forward(self, questions, contexts):
        tokenized = self.tokenizer(list(zip(questions, contexts)),
                                   max_length=self.max_length,
                                   padding=True,
                                   return_overflowing_tokens=True,
                                   return_tensors='pt',
                                   stride=self.stride,
                                   truncation=True).to(self.device)
        overflowing_tokens = tokenized.pop('overflowing_tokens')
        tokenized.pop('num_truncated_tokens')
        outputs = self.model(**tokenized)

        while overflowing_tokens.numel() > 0:
            next_tokenized = self.tokenizer(self.tokenizer.batch_decode(overflowing_tokens),
                                            max_length=self.max_length,
                                            padding=True,
                                            return_overflowing_tokens=True,
                                            return_tensors='pt',
                                            stride=self.stride,
                                            truncation=True).to(self.device)
            overflowing_tokens = next_tokenized.pop('overflowing_tokens')
            next_tokenized.pop('num_truncated_tokens')
            next_outputs = self.model(**next_tokenized)
            outputs.start_logits = torch.cat([outputs.start_logits,
                                              next_outputs.start_logits[:, self.stride:]], dim=-1)
            outputs.end_logits = torch.cat([outputs.end_logits,
                                            next_outputs.end_logits[:, self.stride:]], dim=-1)

        return outputs


class BaselineContextModel(nn.Module):
    """
    The baseline model for the context (sequence classification) task.

    Parameters
    ----------
    device : torch.device
        The device to run the model and inferences on.
    model_class : Hugging Face model with a span classification head on top
        A model such as BertForQuestionAnswering.
    max_length : The maximum number of tokens to use in one pass. (All tokens
        will be covered, since overflow tokens are also passed through the
        model.)
    model_tokenizer : Hugging Face tokenizer
        Tested with non-Rust based tokenizers at present.
    num_contexts : int
        Number of possible context classes to choose from.
    pretrained_model_path : str
        Path to a pretrained model (either in the cloud or on one's machine).
    stride : int
        The amount of overlap between overflow tokens and the tokens already
        passed into the model.
    """

    def __init__(self,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model_class=RobertaForSequenceClassification,
                 max_length=None,
                 model_tokenizer=RobertaTokenizerFast,
                 num_contexts=1024,
                 pretrained_model_path='roberta-large',
                 stride=12):
        super(BaselineContextModel, self).__init__()
        self.device = device
        self.num_contexts = num_contexts
        self.tokenizer = model_tokenizer.from_pretrained(pretrained_model_path)
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length // 2
        self.stride = stride
        self.model = model_class.from_pretrained(pretrained_model_path, num_labels=num_contexts).to(self.device)

    def forward(self, contexts):
        tokenized = self.tokenizer(contexts,
                                   max_length=self.max_length,
                                   padding=True,
                                   return_overflowing_tokens=True,
                                   return_tensors='pt',
                                   stride=self.stride,
                                   truncation=True).to(self.device)
        overflowing_tokens = tokenized.pop('overflowing_tokens')
        tokenized.pop('num_truncated_tokens')
        outputs = self.model(**tokenized)
        print(outputs)

        while overflowing_tokens.numel() > 0:
            next_tokenized = self.tokenizer(contexts,
                                            max_length=self.max_length,
                                            padding=True,
                                            return_overflowing_tokens=True,
                                            return_tensors='pt',
                                            stride=self.stride,
                                            truncation=True).to(self.device)
            overflowing_tokens = next_tokenized.pop('overflowing_tokens')
            next_tokenized.pop('num_truncated_tokens')
            next_outputs = self.model(**next_tokenized)
            outputs.start_logits = torch.cat([outputs.start_logits,
                                              next_outputs.start_logits[:, self.stride:]], dim=-1)
            outputs.end_logits = torch.cat([outputs.end_logits,
                                            next_outputs.end_logits[:, self.stride:]], dim=-1)

        return outputs


if __name__ == '__main__':
    with torch.no_grad():
        model = BaselineQAModel()
        print(model(['How many of the paintings are there?', 'What kind of tests is this?'],
                    ['This many', 'This kind.']))

        model = BaselineContextModel()
        print(model(['This is context 1 that might provide an answer to one kind of question.',
                     'This is context 2 that might provide an answer to another kind of question.']))
