import os
import numpy as np
import pandas as pd
from functools import partial

from datasets import Dataset, Image
from torch.utils.data import DataLoader

def make_dataset(dataset_folder, trial_type=1, manifest_file="manifest.csv"):
    """
    Makes HF Dataset from a given dataset folder and trial type.
    -----
    Inputs: 
    - dataset_folder: the path to the folder containing the dataset
      and manifest
    - trial_type: the trial type of the dataset, indexed by the number of
      images per trial:
        * 0: one text per trial (used for embedding extraction)
        * 1: one image per trial (used for embedding extraction)
        * 2: two images + two texts per trial (used for pairwise matching)
        * 4: four images + one text per trial (used for 4AFC trials)
    - manifest_file: the location of the manifest CSV within the 
      dataset folder; should contain one row per trial, with *relative*
      paths to any images
    Outputs:
    - ds: an object of class Dataset
    """
    manifest = pd.read_csv(os.path.join(dataset_folder, manifest_file))
    for i in range(1, trial_type+1):
        manifest[f"image{i}"] = [dataset_folder+img for img in manifest[f"image{i}"]]
    ds = Dataset.from_pandas(manifest)
    for i in range(1, trial_type+1):
        ds = ds.cast_column(f"image{i}", Image())
    ds.trial_type = trial_type
    return ds

def collator(data):
    """
    Collates data, turning it from a list of dicts to a dict of lists.
    -----
    Inputs: 
    - data: a list of dicts with the same keys for each dict
    Outputs:
    - a dict of lists
    """
    return {k: [ex[k] for ex in data] for k in data[0]}

def trial_collator(data, trial_type):
    """
    Collates data for a single trial, collapsing images and texts.
    -----
    Inputs: 
    - data: data for a single trial, with 2 or 4 images and 2 or 1 texts
    - trial_type: the trial type of the dataset
    Outputs:
    - a dict of images and texts
    """
    image_keys = [f"image{i}" for i in range(1, trial_type+1)]
    text_keys = [f"text{i}" for i in range (1, (trial_type==2)+2)]
    images = [data[0][k] for k in image_keys]
    texts = [data[0][k] for k in text_keys]
    return {"images": images, "texts": texts}

def make_dataloader(dataset):
    """
    Constructs a dataloader from a dataset.
    -----
    Inputs: 
    - dataset: an object of class Dataset, with additional attribute
      trial_type
    Outputs:
    - an object of class Dataloader
    """
    trial_type = dataset.trial_type
    batch_size = 1 if trial_type >= 2 else 16
    collate_fn = partial(trial_collator, trial_type=trial_type) if trial_type >= 2 else collator
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    dl.trial_type = trial_type
    return dl