import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils.preprocessing.data import convert_str_indices_to_token_indices


class ReaderDataset(Dataset):
    """
    Gets training data

    Parameters
    ----------
    qa_dicts : list of dictionaries
        The data structure should look like this:
        [{'answer': 'Project managers, creative types, marketers',
          'context': 'Intake creative project requests tip uses an app ...'
          'mlm': 0.26,  # can be None if not calculated
          'question': 'Who are intake creative project requests useful for?'}, ...]
    fast_tokenizer : A HuggingFace FastTokenizer built upon Rust.
        Used for converting string indices to token indices.
    split : string
        Possible values are 'train' or, if not 'train', then the split
        is considered to be the valid split.
    train_size : float between 0.0 and 1.0
        The size of the train split (validation is the balance remaining).
    """

    def __init__(self,
                 qa_dicts=None,
                 fast_tokenizer=None,
                 split='train',
                 train_size=0.7):
        self.qa_dicts = qa_dicts[:int(len(qa_dicts) * train_size)] if split == 'train' else qa_dicts[int(
            len(qa_dicts) * train_size):]
        self.tokenizer = fast_tokenizer

    def __getitem__(self, item):
        answer = self.qa_dicts[item]['answer']
        context = self.qa_dicts[item]['context'][:2000]  # limiting context for now, but can batch this
        try:
            start_str_index = context.index(answer)
            end_str_index = start_str_index + len(self.qa_dicts[item]['answer'])
        except ValueError:
            start_str_index = 0
            end_str_index = 0
        span_indices = convert_str_indices_to_token_indices(
            self.tokenizer, context,
            [start_str_index, end_str_index], test=False)
        targets = span_indices
        return self.qa_dicts[item]['question'], context, torch.tensor(targets)

    def __len__(self):
        return len(self.qa_dicts)


def dataloader(qa_dicts, fast_tokenizer, batch_size=1, split='train', **kwargs):
    return DataLoader(ReaderDataset(qa_dicts, fast_tokenizer=fast_tokenizer, split=split), batch_size=batch_size,
                      **kwargs)


def base_dataloader(*paths, split=None, **kwargs):
    """
    Loads a base dataloader from `datasets`.

    Parameters
    ----------
    paths : str or sequence of str
        Path(s) pointing to the dataset processing script. Can be a
        dataset identifier in HuggingFace Datasets of a local
        path.
    split : str
        The split of the `dataset` to download (e.g., 'train').
    kwargs : dict
        Additional keyword arguments to pass to `load_dataset`.

    Returns
    -------
    `Dataset` or `DatasetDict`. If `split` is None, a `datasets.DatasetDict`
    is returned. Otherwise, the dataset is returned.
    """
    if split is not None:
        return DataLoader(load_dataset(*paths, split=split, **kwargs))
    return load_dataset(*paths, split=split, **kwargs)


if __name__ == '__main__':
    dataloader = base_dataloader('glue', 'mrpc', split='train')
    for batch in dataloader:
        print(batch)
        break
