from datasets import load_dataset
from torch.utils.data import DataLoader


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
