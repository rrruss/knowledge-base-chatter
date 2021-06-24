import torch
import torch.nn as nn
from transformers import RobertaForQuestionAnswering, RobertaTokenizer, RobertaForSequenceClassification


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
                 model_tokenizer=RobertaTokenizer,
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
                 model_tokenizer=RobertaTokenizer,
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
