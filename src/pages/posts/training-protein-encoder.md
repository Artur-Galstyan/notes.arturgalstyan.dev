---
layout: ../../layouts/PostLayout.astro
title: Training a Protein Encoder (DRAFT)
date: 2026-02-21
---

Protein data usually comes in the form of amino acid (AA) sequences (or 3d coordinates if you're really lucky). But those are just letters. Your computer however only understand numbers. 

So imagine you had a very simple dataset like the TAPE fluorescence dataset. Some proteins can glow in the dark, a discovery which actually lead to a [nobel prize](https://www.nobelprize.org/prizes/chemistry/2008/press-release/). In that dataset you have to predict the `log_fluorescence` of the given protein. 

At this point you have 2 options:

1) Tokenise the protein -> pass into a simple embedding layer -> forward that to a regression head
2) Use pre-trained encoders

Option 2 is especially appealing, because some other company has spend more money than you and I will ever make in our lifetimes to train a model for you. These include models like [ESM2](https://github.com/facebookresearch/esm), [ESMC](https://github.com/evolutionaryscale/esm), [ProtTrans](https://github.com/agemagician/ProtTrans) and likely some others that I'm feeling to lazy to look up right now.

So the idea is then to tokenise the sequence (by using whatever tokenisation the encoder used), pass that through the encoder model and use the embeddings for your downstream task. Usually, you get a matrix back in the shape `[seq_len, embedding_size]` and most of the time you can take the mean across axis 0 to just get a matrix with shape `[embedding_size]` and then pass that to some MLP or something.

But option 2 is boring. We'll just insert another option in there, namely: make our own encoder!

## Option 3: DIY it Yourself

First, some housekeeping. We have 2 RTX5090 at our disposal and only 32 GBs of RAM (and 96 GB of swap lol). 

In terms of training data, we will be training on the UniRef50 dataset. A quick word on UniRef, namely that there are 3 datasets

- UniRef100: every unique sequence (~250M+ sequences) - LOTS of duplicates
- UniRef90: clustered at 90% identity (~150M representatives) - still many duplicates
- UniRef50: clustered at 50% identity (~65M representatives) - every sequence pair is at most 50% similar -> most diversity

Every training point should teach the model something new. If we used UniRef100, we'd constantly be training on duplicate data, which is a waste of time. Thus, we choose the smallest and most diverse of them: UniRef50.

### Data Stuff

Let's write some functions that handle the boring data stuff for us:


<details>
    
```python
import json
import os
import pathlib
import tempfile

import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax.sharding as js
import mlflow
import numpy as np
import optax
from beartype.typing import Any, cast
from datasets import Dataset, load_from_disk
from jaxonlayers.functions.embedding import sinusoidal_embedding
from jaxonlayers.layers import TransformerEncoder
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from jaxonmodels.functions import default_floating_dtype
from jaxonmodels.functions.utils import param_summary
```

<summary>Imports</summary>
</details>

```python
np.random.seed(44)


def setup_mlflow(experiment_name: str = "ProtEmb"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert tracking_uri is not None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


class HFDataSource(grain.RandomAccessDataSource):
    def __init__(self, path):
        self.ds = load_from_disk(path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, record_key):
        row = self.ds[record_key]
        return json.dumps(
            {"sequence": row["sequence"], "length": row["length"]}
        ).encode("utf-8")


def create_datasets_from_fasta(fasta_path, max_seq_len, sizes, output_dir="data"):
    sequences = []
    current_seq = []

    existing_paths = [
        pathlib.Path(f"{output_dir}/uniref50_{size // 1000}k").exists()
        for size in sizes
    ]
    if all(existing_paths):
        return

    for line in open(fasta_path, "rt"):
        line = line.strip()
        if line.startswith(">"):
            seq = "".join(current_seq)
            if 0 < len(seq) <= max_seq_len:
                sequences.append(seq)
            current_seq = []
        else:
            current_seq.append(line)

    seq = "".join(current_seq)
    if 0 < len(seq) <= max_seq_len:
        sequences.append(seq)

    for size in sizes:
        if size is None:
            label = "full"
            subset = sequences
        else:
            label = f"{size // 1000}k"
            subset = sequences[:size]

        path = f"{output_dir}/uniref50_{label}"
        if pathlib.Path(path).exists():
            continue

        ds = Dataset.from_dict(
            {
                "sequence": subset,
                "length": [len(s) for s in subset],
            }
        )
        ds.save_to_disk(path)
        print(f"Saved {len(subset)} sequences to {path}")


def setup_data():
    sizes = [10_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, None]
    create_datasets_from_fasta(
        "uniref50.fasta",
        max_seq_len=1024,
        sizes=sizes,
    )

    for size in sizes:
        dataset_name = f"data/uniref50_{size // 1000}k"
        if pathlib.Path(f"{dataset_name}/train").exists():
            continue
        ds = load_from_disk(dataset_name)
        train_rest = ds.train_test_split(test_size=0.2, seed=42)  # ty:ignore[unresolved-attribute]
        val_test = train_rest["test"].train_test_split(test_size=0.5, seed=42)
        train_rest["train"].save_to_disk(f"{dataset_name}/train")
        val_test["train"].save_to_disk(f"{dataset_name}/val")
        val_test["test"].save_to_disk(f"{dataset_name}/test")
        print(
            f"Split {dataset_name}: train={len(train_rest['train'])}, "
            f"val={len(val_test['train'])}, test={len(val_test['test'])}"
```

A couple of things are happening here. As you can see, we're splitting the UniRef50 dataset into subsets for fast iteration and prototyping. We'll be using `grain` as our dataloader and `mlflow` to track our experiments. In terms of the `max_seq_len` (which will come in just a moment) I chose 1024. Except for [Titin](https://en.wikipedia.org/wiki/Titin), which can have a length between 27,000 and 35,000 AA, I don't know of any other protein is longer than 1024. I think with 1024, we have probably 90% of the proteins covered (but I have no proof here, this is just speculation).

But now that we have our data, let's forget about this code as we will likely never touch it again. 

The next thing we need is a tokeniser.

```python
class Tokenizer:
    _amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    _special_tokens = ["<mask>", "<cls>", "<eos>", "<unk>"]

    vocab = {
        "<pad>": 0,
        **{aa: i + 1 for i, aa in enumerate(_amino_acids)},
        **{st: i + 21 for i, st in enumerate(_special_tokens)},
    }
    inverse_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def encode(cls, sequence: list[str]) -> list[int]:
        return [cls.vocab.get(aa, cls.vocab["<unk>"]) for aa in sequence]

    @classmethod
    def decode(cls, encoded: list[int]) -> list[str]:
        return [cls.inverse_vocab[token] for token in encoded]

    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    EOS_TOKEN = "<eos>"
    MASK_TOKEN = "<mask>"
    MASK_ID = vocab["<mask>"]
    PAD_ID = 0
    VOCAB_SIZE = len(vocab)
```

This is a very simple one. It maps the AA to an integer and we also have a few special tokens, the most important ones are the `PAD_TOKEN` and the `MASK_TOKEN`. We also have the `CLS_TOKEN` (start of sequence) and the `EOS_TOKEN` (end of sequence). An example of what the model might be trained on could look like this:


```
<pad> <pad> <pad> <pad> <pad> [...] <cls> A C D E A <mask> A <mask> <eos>
```

The goal of the model is to predict what AA goes into the mask tokens. This is called _masked language modelling_ or MLM and is the most basic and standard way to train an encoder.
