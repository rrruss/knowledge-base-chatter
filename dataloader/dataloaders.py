from datasets import load_dataset


def base_dataloader(path, **kwargs):
    """
    Path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
              e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``.
            - a dataset identifier in the HuggingFace Datasets Hub (list all available datasets and ids with ``datasets.list_datasets()``)
              e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``.
    Parameters
    ----------
    path : str
        Path pointing to the dataset processing script. Can be a
        dataset identifier in HuggingFace Datasets of a local
        path.
    kwargs : dict
        Additional keyword arguments to pass to `load_dataset`.

    Returns
    -------
    `Dataset` or `DatasetDict`. If `split` is None, a `datasets.DatasetDict`
    is returned. Otherwise, the dataset is returned.
    """
    return load_dataset(path, **kwargs)
