from datasets import load_dataset
import os

def load_books():
    dataset = load_dataset(
        "VsquareSheremetyevo/goodreads_books",
        split="train"
    )
    return dataset.to_pandas()
