---
layout: ../../layouts/PostLayout.astro
title: Hello, Proteins!
date: 2025-09-07
---

While language, image and video models are pretty cool (and they really are!), I think one of the best application for AI (until we discover AGI) is in drug discovery. And in the realm of drug discovery, proteins are **key**!

## Proteins, TL;DR

Proteins are little molecular machines, that do *something*, depending on their *shape*. For proteins, *shape* is **everything**! If you know what the proteins looks like in 3D, you can predict what they can do or how they interact with other molecules. That's what modern research such as AlphaFold are able to do.

To make a protein, you select a particular section of the DNA (called a *gene*) and *transcribe* it to a molecule called messenger RNA (mRNA), which moves out of the nucleus of the cell (if it had one) to a different molecule called a ribosome. The ribosome takes the mRNA and constructs the protein (this is called translation).

And voilà, you have a protein.

So, TL;DR: gene -> transcribe into -> mRNA -> move to ribosome -> translate mRNA into a sequence of amino acids (which is what a protein basically is). This is called the *central dogma of biology*.

## Hello, ~World~ Proteins!

In "normal" machine learning, the MNIST dataset is the equivalent of a hello world program and for proteins, I think, the solubility prediction is their version of it.

Some amino acids are more hydrophilic (water loving) than others (hydrophobic), which is important for proteins whether or not they disolve in water or not. This becomes especially relevant when you think about some drugs that you want to get *into* the body; they need to disolve somehow.

There is a dataset on Huggingface for that, that you can find [here](https://huggingface.co/datasets/proteinea/solubility). Here's what it looks like:

You have a sequence of amino acids, like this one:

```
GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEEMRKEIIAVIQKYTKSSDIHFKTLDSNQSVETIEVEIILPR
```

and the target (label) is 1, meaning it is soluble. Each letter in the string is an amino acid. By the way, each animo acid is equal in their "backbone", a collection of atoms and bonds. They only differ on their side-chains (called the R-group), which is attached to the alpha-carbon in the backbone. This side-chain determines the properties of the amino acid. So, to differentiate the amino acids, each letter essentially refers to a "different side-chain".

Ok, so what can we do here? First, we need to explore the data a bit and perhaps find some patterns. Perhaps, we can determine if it is soluble based on the frequency of certain amino acids, along the lines of "more hydrophilic amino acids equals solubility". Let's see:

![Solubility vs. Frequency](/posts/hello-proteins/Figure_1.png)

Hmm, this doesn't really tell us anything. The reason for that is that proteins are not 1D strings, they are 3D structures. You could have a bunch of hydrophilic acids in the center of the protein, surrounded by a thin layer of hydrophobic acids, making the whole protein hydrophobic.

So, let's start simple and create a very basic baseline: a simple linear model that gets these frequencies as input. Here it is:

<details>
    <summary>The first version</summary>

```python
import os
import pickle

import clu.metrics as clum
import equinox as eqx
import flax
import grain
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import polars as pl
import tensorflow_datasets as tfds
from tqdm import tqdm


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")  # pyright: ignore
    accuracy: clum.Average.from_output("accuracy")  # pyright: ignore


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab = ["Z"] + list(amino_acids)
int_to_char = {i: char for i, char in enumerate(vocab)}
char_to_int = {char: i for i, char in enumerate(vocab)}


class SimpleModel(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, key: jt.PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        hidden_layers = 32
        self.linear1 = eqx.nn.Linear(in_features, hidden_layers, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_layers, out_features, key=k2)

    def __call__(self, x: jt.Array) -> jt.Array:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


def loss_fn(model: SimpleModel, x: jt.Array, y: jt.Array) -> tuple[jt.Array, jt.Array]:
    preds = eqx.filter_vmap(model)(x)
    return jnp.mean((preds - y) ** 2), preds


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[SimpleModel, optax.OptState, dict]:
    print("step_fn JIT")
    (loss_value, preds), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    rounded_preds = jnp.round(preds)
    accuracy = jnp.mean(rounded_preds == y)

    metrics = {"loss": loss_value, "accuracy": accuracy}
    return model, opt_state, metrics


@eqx.filter_jit
def eval_step(model: SimpleModel, x: jt.Array, y: jt.Array):
    print("eval_step JIT")
    preds = eqx.filter_vmap(model)(x)
    loss = jnp.mean((preds - y) ** 2)
    rounded_preds = jnp.round(preds)
    correct_preds = jnp.sum(rounded_preds == y)
    return loss, correct_preds


def eval_fn(model: SimpleModel, loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data in loader:
        x = data["features"]
        y = data["label"]
        num_samples = x.shape[0]
        batch_loss, batch_correct = eval_step(model, x, y)

        total_loss += batch_loss * num_samples
        total_correct += batch_correct
        total_samples += num_samples

    if total_samples == 0:
        return {"loss": float("inf"), "accuracy": 0.0}

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]

        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")

        sequence_length = len(sequence)
        frequencies = []

        for acid in vocab:
            acid_count = sequence.count(acid)
            frequencies.append(acid_count / sequence_length)

        return {
            "features": jnp.array(frequencies, dtype=jnp.float32),
            "label": jnp.array(label),
        }


batch_size = 128
builder = tfds.dataset_builders.CroissantBuilder(
    jsonld="https://huggingface.co/api/datasets/proteinea/solubility/croissant",
    file_format="array_record",
)
builder.download_and_prepare()

train_source, test_source = builder.as_data_source(split=["train[:80%]", "train[80%:]"])

dataset_size = len(train_source)
steps_per_epoch = dataset_size // batch_size
print(f"Dataset size: {dataset_size}")
print(f"Steps per epoch: {steps_per_epoch}")

train_index_sampler = grain.samplers.IndexSampler(
    num_records=dataset_size,
    shuffle=True,
    shard_options=grain.sharding.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True
    ),
    seed=4,
    num_epochs=1,
)

train_data_loader = grain.DataLoader(
    data_source=train_source,  # pyright: ignore
    operations=[ProteinPreprocessor(), grain.transforms.Batch(batch_size=batch_size)],
    sampler=train_index_sampler,
)

n_epochs = 10
train_metrics = LossMetrics.empty()


in_features = len(vocab)
out_features = 1

model = SimpleModel(in_features, out_features, key=jax.random.key(42))
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


train_metrics = LossMetrics.empty()
test_metrics = LossMetrics.empty()


for epoch in range(n_epochs):
    train_index_sampler = grain.samplers.IndexSampler(
        num_records=dataset_size,
        shuffle=True,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=4 + epoch,
        num_epochs=1,
    )

    train_data_loader = grain.DataLoader(
        data_source=train_source,  # pyright: ignore
        operations=[
            ProteinPreprocessor(),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=train_index_sampler,
    )

    epoch_steps = 0
    epoch_train_metrics = LossMetrics.empty()
    for data in tqdm(train_data_loader, total=steps_per_epoch, desc=f"Epoch {epoch}"):
        x = data["features"]
        y = data["label"]
        model, opt_state, step_metrics = step_fn(model, x, y, optimizer, opt_state)
        epoch_train_metrics = epoch_train_metrics.merge(
            LossMetrics.single_from_model_output(
                loss=step_metrics["loss"], accuracy=step_metrics["accuracy"]
            )
        )
        epoch_steps += 1

        if epoch_steps >= steps_per_epoch:
            break

    test_dataset_size = len(test_source)
    test_steps = test_dataset_size // batch_size

    test_index_sampler = grain.samplers.IndexSampler(
        num_records=test_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )

    test_data_loader = grain.DataLoader(
        data_source=test_source,  # pyright: ignore
        operations=[
            ProteinPreprocessor(),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=test_index_sampler,
    )

    eval_metrics = eval_fn(model, test_data_loader)

    print(f"Epoch: {epoch}")
    print(f"Train - {epoch_train_metrics.compute()}")
    print(
        f"Test  - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}"
    )
```
</details>

This code is extremely simple: we use `tensorflow_datasets` to fetch the dataset and get it as a data source using the `CroissantBuilder` and to split it to train and test sets. Then we use `grain` as a dataloader and sampler and apply the `ProteinPreprocessor` mapping to the data, which calculates the frequencies of the amino acids. Our very simple, 2 layer network gets those as input and calculates the MSE and uses the `adam` optimiser. The NN library of choice is of course Equinox. As a bonus, we use `clu` to keep track of some metrics. At this point, we could also add `mlflow` to track our experiments, but I'm feeling a bit lazy now and the experiments aren't really that big as of now. We let the model train for 10 epochs and this is the end result:

```
Train - {'loss': Array(0.24376544, dtype=float32), 'accuracy': Array(0.5820112, dtype=float32)}
Test  - Loss: 0.2429, Accuracy: 74.6300
```

This is our baseline. Can we improve on this?

## Second Attempt

Perhaps, using the frequencies was not the right method. We can look towards LLMs for inspiration, because what we have here at hand is kind of a language, just that the vocabulary consists of 20 letters (actually 21 if we count the "padding" acid). So, what we can do here is to use embeddings!

We have a sequence of letters of different lengths. So first we have to choose a `max_seq_len` (those coming from LLM research will be very familiar with this term). We could use any arbitrary number but a better method might be to take the $n$th percentile length, i.e. a length, which covers $n%$ of the entire dataset. A good value for $n$ might be e.g. 90, which would mean that we take a length under which 90% of all the sequences fall under. Those that are longer are simply truncated and those that are shorter get padded with a "special amino acid" `Z` (which is not a letter assigned for amino acids, hence a "free letter"). Furthermore, we need to turn the letters into integers so that we can embed them.

We can do this all using this preprocessor:

```python

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab = ["Z"] + list(amino_acids)
int_to_char = {i: char for i, char in enumerate(vocab)}
char_to_int = {char: i for i, char in enumerate(vocab)}

splits = {
    "train": "solubility_training.csv",
    "validation": "solubility_validation.csv",
    "test": "solubility_testing.csv",
}
train_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["train"])
test_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["test"])
validation_df = pl.read_csv(
    "hf://datasets/proteinea/solubility/" + splits["validation"]
)
PERCENTILE = 90

train_df = train_df.with_columns(
    pl.col("sequences").str.len_chars().alias("length"),
)
max_protein_length = int(
    train_df.select(pl.col("length").quantile(PERCENTILE / 100)).item()
)


class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[:max_protein_length]
        sequence = sequence.ljust(max_protein_length, "Z")
        indices = [char_to_int[aa] for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "label": jnp.array(label),
        }
```

From here, we can now use embeddings and see if this improves our model.

We need to make a few tweaks to our model first though:

```python
class SimpleModel(eqx.Module):
    embedding: eqx.nn.Embedding
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    linear: eqx.nn.Linear

    def __init__(
        self, n_vocab: int, embedding_size: int, out_features: int, key: jt.PRNGKeyArray
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.embedding = eqx.nn.Embedding(n_vocab, embedding_size, key=k1)
        self.conv1 = eqx.nn.Conv1d(embedding_size, 128, kernel_size=7, key=k2)
        self.conv2 = eqx.nn.Conv1d(128, 128, kernel_size=5, key=k3)
        self.linear = eqx.nn.Linear(128, out_features, key=k4)

    def __call__(self, x: jt.Int[jt.Array, " seq_len"]) -> jt.Array:
        x = eqx.filter_vmap(self.embedding)(x)
        x = jnp.transpose(x)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = jnp.max(x, axis=-1)
        x = self.linear(x)
        return x
```

The main change is that we have now an embedding layer and use `Conv1d` layers to model sequence data. We'll get to the more exciting stuff like RNNs or transformers soon enough, don't worry.

Our model didn't really improve however:

```
Train - {'loss': Array(0.24375229, dtype=float32), 'accuracy': Array(0.5818594, dtype=float32)}
Test  - Loss: 0.2431, Accuracy: 74.6787
```

## Little Whoopsie

At this point, I was a bit confused and especially skeptical about the high accuracy on the test data, but pathetic performance on the training data, as it's usually the other way around.

The issue was the model, the used loss function as well as the general problem at hand. You see, we are dealing with a classification problem (0 or 1, soluble or not), but our model was set up for a regression task -> i.e. it outputs a continious number instead of a class.

To remedy this, we change the loss function to this:

```python
def loss_fn(model: SimpleModel, x: jt.Array, y: jt.Array) -> tuple[jt.Array, jt.Array]:
    logits = eqx.filter_vmap(model)(x)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y)), logits
```

We also need to change the preprocessor to one-hot encode the labels:

```python
class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[:max_protein_length]
        sequence = sequence.ljust(max_protein_length, "Z")
        indices = [char_to_int[aa] for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }
```

Also, with this change we have to correct the way the accuracy is calculated:

```python
@eqx.filter_jit
def eval_step(model: SimpleModel, x: jt.Array, y: jt.Array):
    print("eval_step JIT")
    logits = eqx.filter_vmap(model)(x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    predicted_classes = jnp.argmax(logits, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    correct_preds = jnp.sum(predicted_classes == true_classes)

    return loss, correct_preds


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[SimpleModel, optax.OptState, dict]:
    print("step_fn JIT")
    (loss_value, preds), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    predicted_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    accuracy = jnp.mean(predicted_classes == true_classes)

    metrics = {"loss": loss_value, "accuracy": accuracy}
    return model, opt_state, metrics
```

and change `output_features` to be `2`. With that done, the "correct" loss is this:

```
Train - {'loss': Array(0.5717863, dtype=float32), 'accuracy': Array(0.63341343, dtype=float32)}
Test  - Loss: 0.5651, Accuracy: 0.6384
```

And this makes just WAY more sense.

## Some Model Parameter Changes

I tried a couple of configurations, such as these hyperparameters

```python
out_features = 2
embedding_size = 16
learning_rate = 0.001
n_epochs = 10
```

and this model

```python
class SimpleModel(eqx.Module):
    embedding: eqx.nn.Embedding
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    linear: eqx.nn.Linear

    def __init__(
        self, n_vocab: int, embedding_size: int, out_features: int, key: jt.PRNGKeyArray
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        hidden_size = 32
        self.embedding = eqx.nn.Embedding(n_vocab, embedding_size, key=k1)
        self.conv1 = eqx.nn.Conv1d(embedding_size, hidden_size, kernel_size=7, key=k2)
        self.conv2 = eqx.nn.Conv1d(hidden_size, hidden_size, kernel_size=5, key=k3)
        self.linear = eqx.nn.Linear(hidden_size, out_features, key=k4)

    def __call__(self, x: jt.Int[jt.Array, " seq_len"]) -> jt.Array:
        x = eqx.filter_vmap(self.embedding)(x)
        x = jnp.transpose(x)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = jnp.max(x, axis=-1)
        x = self.linear(x)
        return x
```

but my loss seems to plateau at this value:

```
Train - {'loss': Array(0.5040968, dtype=float32), 'accuracy': Array(0.7300881, dtype=float32)}
Test  - Loss: 0.5948, Accuracy: 0.6587
```

regardless of the hyperparameters. This means, I need a better model.

## The Crossroad

Ok so at this point, I have a couple of options. My model is bad because it doesn't really understand that proteins fold. It's working with a 1D string unaware of what it actually means in the physical world. I need to TELL it that somehow.

We can once again look at LLM research for guidance. There we have models called "encoders". Those take in some text and output a vector/matrix in latent space. The idea being that the encoded vector carries more information about the protein sequence than our crude `embeddings` layer.

In the world of language, a word like "king" becomes a vector that points in some direction and the word "man" becomes a vector that might point to a similar direction, because they are semantically similar (a king is a man). The word "giraffe" would point in a completely different direction (it has nothing really in common with "king" or "man"). So those vectors have "meaning" (at least in relation to each other). My idea is that I could use those encoded vectors as features in my model.

Ok, but where do I get those vectors from? Luckily, there is a company that has already pretrained such a model and made it available for everyone. The model in question is called `esm-c`, which is the encoder model in their model familiy. In those vectors, I'm hoping that proteries from how they are folded are in there that my model head can then pick up.

Honestly, I don't really know how encoders are trained (yet!) and for now I will treat this as a black box, mainly to establish an upper bound. AFAIK, it's got something to do with masked language modelling, i.e. _fill in the blank_ kind of training. I can't compete against that company with their many PhDs working round the clock - I have 1/10th their brains and $1e^{-18}$ of their resources. This means, I will use their encoder, hopefully get some nice performance and then _try_ to create my own tiny encoder model and see how close I can get.

## The Sad Part

The sad part is that the ESM-C model is written in ... you guessed it PyTorch! _Surprised Pikachu Face_.

This complicates things a bit and basically means that the encoder is _frozen_ to me. My gradients won't flow through the encoder and I won't be able to update it - it's not differentiable for me. But do I even need to? The answer is probably no, because I won't find proteins on the internet (or in the wild, even if I had a lab) that they haven't trained the model on already. This means that I don't have any data to really improve it.

Ok but the model is in PyTorch and my model is in JAX, so how do I connect the two? Well there are 2 ways:

### The CPU Round Trip
This is the most obvious solution. PyTorch computed some tensor, you do a little `to_cpu()` or whatever the function is in PyTorch, then maybe turn that into a Numpy array, and then finally pass that Numpy array into a JAX array. But I don't think I need to tell you that this is very slow. You might have PyTorch tensors that are multiple gigabytes large which means you are wasting precious GPU bandwith and CPU power for the conversion. But the tensor is already in the GPU, the numbers are there. The GPU itself doesn't care about the framework or something; e.g. for NVIDIA, it's all just CUDA wrappers anyway. Enter the second way.

### dlpack

Researchers and engineers have notices this issue and came up with a solution, a standard format, that allows each framework (if they are compatible) to just "look up" those numbers in the GPU (i.e. use pointers) and say whippety whoppety, these numbers are now my property. More technically, you export the data as a dlpack capsule, which contains pointers to the actual numbers and some metadata describing the tensor. dlpack compatible frameworks (such as JAX or PyTorch) can take these capsules and "import"  (and export) them. This is what we will do.


## Some Thoughts as the Model is Training

I have added the ESM-C model into the pipeline and changed our model to be a simple MLP. It gets as input the output of the ESM encoder but with mean pooling applied to it (because the output of the encoder is of shape 460x960, where 460 is the sequence length and 960 is the embedding size).

Currently, as of writing this, I'm in Switzerland and only have my laptop with me. I generated all the embeddings (which quickly became a folder > 120 GBs) and now I'm simply loading those in as needed. But I only have my laptop. A single epoch takes around 5 hours on a CPU, which is super duper slow (mainly because of I/O but what can you do).

But as the model is training, I was thinking about different encoder models. Yes, I will train a BERT-like model (similar to ESM) but I might also try out other wild ideas (like SSMs for encoders).

It was here that I noticed that encoders are trained _differently_, i.e. on different objectives. BERT-like encoders train on masked language modelling while SSMs are trained on god-knows-what. So there is no **direct** way to compare these models.

However, we can use our solubility task (the so-called _downstream task_) as a proxy! The higher the performance on the downstream task the better (for that downstream task) is the encoder.

This doesn't mean that my stitched-up Frankenstein's model is always better and to measure the general performance, we actually need more downstream tasks and measure the mean performance delta across all tasks. For now, it's just solubility, but this is good to keep in mind for later.


## Actually Training the Thing

Ok, so it's been a couple of days, but here's the current code:

<details>
    <summary>Data Loading and Preprocessing</summary>

```python
import json
import os

import array_record
import grain
import jax.numpy as jnp
import polars as pl
from Bio import SeqIO

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = ["Z"] + list(AMINO_ACIDS)
INT_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}


class ProteinPreprocessor(grain.transforms.Map):
    max_protein_length: int

    def __init__(self, max_protein_length: int):
        self.max_protein_length = max_protein_length

    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[: self.max_protein_length]
        sequence = sequence.ljust(self.max_protein_length, "Z")
        indices = [CHAR_TO_INT.get(aa, 0) for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "sequence": sequence,
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }


class DataFrameDataSource:
    def __init__(self, df: pl.DataFrame):
        self._data = df.to_dicts()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, record_key: int) -> dict:
        return self._data[record_key]


def parse_plmsol_fasta(fasta_path):
    sequences = []
    solubility_labels = []
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File {fasta_path} not found.")
    for record in SeqIO.parse(open(fasta_path), "fasta"):
        try:
            solubility_str = record.description.split(" ")[-1].split("-")[-1]
            sequences.append(str(record.seq))
            solubility_labels.append(solubility_str)
        except IndexError:
            print(f"Warning: Could not parse label from: {record.description}")
            continue
    return sequences, solubility_labels


def convert_plmsol_labels_to_binary(labels):
    binary_labels = []
    for label in labels:
        if label == "U":
            binary_labels.append(0)
        elif label == "0":
            binary_labels.append(0)
        elif label == "1":
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels


def write_df_to_json_array_record(df, path):
    with array_record.ArrayRecordWriter(path, "w") as writer:
        for record in df.to_dicts():
            record_bytes = json.dumps(record).encode("utf-8")
            writer.write(record_bytes)


def get_datasources(percentile: int):
    proteinea_splits = {
        "train": "solubility_training.csv",
        "validation": "solubility_validation.csv",
        "test": "solubility_testing.csv",
    }

    plmsol_base_path = "experiments/plmsol_dataset"
    plmsol_splits = {
        "train": "train_dataset.fasta",
        "validation": "validation_dataset.fasta",
        "test": "test_dataset.fasta",
    }

    proteinea_train_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["train"]
    )
    proteinea_test_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["test"]
    )
    proteinea_validation_df = pl.read_csv(
        "hf://datasets/proteinea/solubility/" + proteinea_splits["validation"]
    )

    plmsol_dfs = {}
    for split_name, file_name in plmsol_splits.items():
        file_path = os.path.join(plmsol_base_path, file_name)
        seqs, string_labels = parse_plmsol_fasta(file_path)
        if seqs:
            binary_labels = convert_plmsol_labels_to_binary(string_labels)
            plmsol_dfs[split_name] = pl.DataFrame(
                {"sequences": seqs, "labels": binary_labels}
            )
        else:
            plmsol_dfs[split_name] = pl.DataFrame(
                {"sequences": [], "labels": []},
                schema={"sequences": pl.String, "labels": pl.Int64},
            )

    plmsol_train_df = plmsol_dfs.get("train")
    plmsol_test_df = plmsol_dfs.get("test")
    plmsol_validation_df = plmsol_dfs.get("validation")

    assert plmsol_train_df is not None
    assert plmsol_test_df is not None
    assert plmsol_validation_df is not None

    train_df = pl.concat([proteinea_train_df, plmsol_train_df])
    test_df = pl.concat([proteinea_test_df, plmsol_test_df])
    validation_df = pl.concat([proteinea_validation_df, plmsol_validation_df])

    train_df = train_df.with_columns(
        pl.col("sequences").str.len_chars().alias("length"),
    )
    max_protein_length = int(
        train_df.select(pl.col("length").quantile(percentile / 100)).item()
    )

    train_source = DataFrameDataSource(train_df)
    test_source = DataFrameDataSource(test_df)
    validation_source = DataFrameDataSource(validation_df)

    return train_source, test_source, validation_source, max_protein_length


def get_dataloaders(
    train_source,
    test_source,
    validation_source,
    epoch: int,
    batch_size: int,
    max_protein_length: int,
):
    train_index_sampler = grain.samplers.IndexSampler(
        num_records=len(train_source),
        shuffle=True,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=4 + epoch,
        num_epochs=1,
    )
    train_data_loader = grain.DataLoader(
        data_source=train_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=train_index_sampler,
    )

    test_dataset_size = len(test_source)
    validation_dataset_size = len(validation_source)
    test_index_sampler = grain.samplers.IndexSampler(
        num_records=test_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )
    test_data_loader = grain.DataLoader(
        data_source=test_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=test_index_sampler,
    )
    validation_index_sampler = grain.samplers.IndexSampler(
        num_records=validation_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )
    validation_data_loader = grain.DataLoader(
        data_source=validation_source,
        operations=[
            ProteinPreprocessor(max_protein_length),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=validation_index_sampler,
    )

    return train_data_loader, test_data_loader, validation_data_loader
```
</details>


<details>
    <summary>The Training Loop</summary>

```python
import os

import clu.metrics as clum
import equinox as eqx
import flax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mlflow
import optax
import torch
from data import get_dataloaders, get_datasources
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tabulate import tabulate
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RUN_DESCRIPTION = "ESM-C 300M embeddings + simple backbone (no attention)"

# Global ESM client
ESM_CLIENT = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"


class SimpleModel(eqx.Module):
    mlp: eqx.nn.MLP
    linear: eqx.nn.Linear

    def __init__(
        self,
        embedding_size: int,
        out_features: int,
        key: jt.PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key, 2)
        d_inner = 128

        self.mlp = eqx.nn.MLP(
            embedding_size, d_inner, width_size=d_inner, depth=4, key=k1
        )
        self.linear = eqx.nn.Linear(d_inner, out_features, key=k2)

    def __call__(
        self,
        x: jt.Float[jt.Array, "seq_len embed_dim"],
        state: eqx.nn.State,
        key,
        inference,
    ) -> tuple[jt.Array, eqx.nn.State]:
        x = jnp.mean(x, axis=0)
        x = self.mlp(x)
        x = self.linear(x)
        return x, state


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")
    accuracy: clum.Average.from_output("accuracy")


def loss_fn(
    model: SimpleModel,
    state: eqx.nn.State,
    x: jt.Array,
    y: jt.Array,
    key: jt.PRNGKeyArray | None,
    inference: bool,
) -> tuple[jt.Array, tuple[jt.Array, eqx.nn.State]]:
    if inference is False and key is not None:
        keys = jax.random.split(key, x.shape[0])
        in_axes = (0, None, 0, None)
    else:
        keys = key
        in_axes = (0, None, None, None)

    logits, state = eqx.filter_vmap(
        model, in_axes=in_axes, out_axes=(0, None), axis_name="batch"
    )(x, state, keys, inference)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y)), (
        logits,
        state,
    )


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    state: eqx.nn.State,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jt.PRNGKeyArray,
) -> tuple[SimpleModel, eqx.nn.State, optax.OptState, dict]:
    inference = False
    (loss_value, (preds, state)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, state, x, y, key, inference)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    predicted_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    accuracy = jnp.mean(predicted_classes == true_classes)
    metrics = {"loss": loss_value, "accuracy": accuracy}
    return model, state, opt_state, metrics


@eqx.filter_jit
def eval_step(model: SimpleModel, state: eqx.nn.State, x: jt.Array, y: jt.Array):
    key, inference = None, True
    logits, state = eqx.filter_vmap(
        model, in_axes=(0, None, None, None), out_axes=(0, None), axis_name="batch"
    )(x, state, key, inference)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))
    predicted_classes = jnp.argmax(logits, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    correct_preds = jnp.sum(predicted_classes == true_classes)
    return loss, correct_preds


def eval_fn(model: SimpleModel, state: eqx.nn.State, loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for data in loader:
        xs = []
        for seq in data["sequence"]:
            protein = ESMProtein(sequence=seq)
            protein_tensor = ESM_CLIENT.encode(protein)
            logits_output = ESM_CLIENT.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            assert logits_output.embeddings is not None
            embedding = logits_output.embeddings.squeeze(1)
            embedding = embedding.cpu().numpy()
            xs.append(embedding)
        x = torch.stack(xs)
        x = jax.dlpack.from_dlpack(x)
        y = data["label"]

        num_samples = x.shape[0]
        batch_loss, batch_correct = eval_step(model, state, x, y)
        total_loss += batch_loss * num_samples
        total_correct += batch_correct
        total_samples += num_samples
    if total_samples == 0:
        return {"loss": float("inf"), "accuracy": 0.0}
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


def train(
    n_epochs: int,
    steps_per_epoch: int,
    train_source,
    test_source,
    validation_source,
    batch_size: int,
    model: SimpleModel,
    state: eqx.nn.State,
    learning_rate: float,
    max_protein_length: int,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.key(12)

    results_table = []

    with mlflow.start_run(description=RUN_DESCRIPTION):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "model": str(model),
                "max_protein_length": max_protein_length,
            }
        )

        for epoch in range(n_epochs):
            train_loader, test_loader, validation_loader = get_dataloaders(
                train_source,
                test_source,
                validation_source,
                epoch,
                batch_size,
                max_protein_length,
            )

            epoch_steps = 0
            epoch_train_metrics = LossMetrics.empty()
            pbar = tqdm(
                train_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for data in pbar:
                xs = []
                for seq in data["sequence"]:
                    protein = ESMProtein(sequence=seq)
                    protein_tensor = ESM_CLIENT.encode(protein)
                    logits_output = ESM_CLIENT.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True),
                    )
                    assert logits_output.embeddings is not None
                    embedding = logits_output.embeddings.squeeze(1).squeeze(0)
                    xs.append(embedding)
                x = torch.stack(xs)
                x = jax.dlpack.from_dlpack(x)
                y = data["label"]
                key, step_key = jax.random.split(key)
                model, state, opt_state, step_metrics = step_fn(
                    model, state, x, y, optimizer, opt_state, step_key
                )
                epoch_train_metrics = epoch_train_metrics.merge(
                    LossMetrics.single_from_model_output(
                        loss=step_metrics["loss"], accuracy=step_metrics["accuracy"]
                    )
                )
                epoch_steps += 1
                computed_metrics = epoch_train_metrics.compute()
                pbar.set_postfix(
                    loss_and_acc=f"Loss: {computed_metrics['loss']:.4f}, Accuracy: {computed_metrics['accuracy']:.4f}"
                )
                if epoch_steps >= steps_per_epoch:
                    break

            epoch_eval_metrics = eval_fn(model, state, test_loader)
            epoch_validation_metrics = eval_fn(model, state, validation_loader)

            train_metrics = epoch_train_metrics.compute()

            results_table.append(
                [
                    epoch,
                    f"{float(train_metrics['loss']):.4f}",
                    f"{float(train_metrics['accuracy']):.4f}",
                    f"{float(epoch_eval_metrics['loss']):.4f}",
                    f"{float(epoch_eval_metrics['accuracy']):.4f}",
                    f"{float(epoch_validation_metrics['loss']):.4f}",
                    f"{float(epoch_validation_metrics['accuracy']):.4f}",
                ]
            )

            print(
                "\n"
                + tabulate(
                    results_table,
                    headers=[
                        "Epoch",
                        "Train Loss",
                        "Train Acc",
                        "Test Loss",
                        "Test Acc",
                        "Val Loss",
                        "Val Acc",
                    ],
                    tablefmt="simple_grid",
                )
            )

            mlflow.log_metrics(
                {
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "test_loss": float(epoch_eval_metrics["loss"]),
                    "test_accuracy": float(epoch_eval_metrics["accuracy"]),
                    "validation_loss": float(epoch_validation_metrics["loss"]),
                    "validation_accuracy": float(epoch_validation_metrics["accuracy"]),
                },
                step=epoch,
            )

    return model, opt_state


def main():
    print("Experiment:", RUN_DESCRIPTION)
    experiment_name = "protein_solubility_classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.enable_system_metrics_logging()

    out_features = 2
    esm_embedding_dim = 960
    learning_rate = 3e-4
    n_epochs = 50
    batch_size = 128
    percentile = 90

    train_source, test_source, validation_source, max_protein_length = get_datasources(
        percentile
    )

    dataset_size = len(train_source)
    steps_per_epoch = dataset_size // batch_size

    model, state = eqx.nn.make_with_state(SimpleModel)(
        esm_embedding_dim,
        out_features,
        key=jax.random.key(42),
    )

    train(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        train_source=train_source,
        test_source=test_source,
        validation_source=validation_source,
        batch_size=batch_size,
        model=model,
        state=state,
        learning_rate=learning_rate,
        max_protein_length=max_protein_length,
    )


if __name__ == "__main__":
    main()
```

</details>


Our model is pretty simple and it just gets the embeddings from the ESM model. Interestingly, as I was writing this, JAX has updated the implementation for the `dlpack` module. Now, it does **not** support dlpack capsules anymore. At first, I was annoyed by this, but it turns out, that the new way is much simpler: just pass the PyTorch tensor directly to the dlpack module, e.g.

```python
x = torch.stack(xs)
x = jax.dlpack.from_dlpack(x)
```

And this is pretty neat!


## Some Thoughts During the Training

I actually played around with a couple of different architectures (e.g. I tried Mamba as an encoder model) and often I would get good performance on the training set, but it'd get pretty bad on the testing and validation data and start to overfit. The best I could do on test and validation was around 65% at most, which is just a bit better than randomly guessing. Turns out, protein solubility prediction is not as easy as I thought!

I also got my hands on more proteins! I was initially using the proteins from the [proteina/solubility](hf://datasets/proteinea/solubility/) dataset from Huggingface, but then found more proteins from [PLM_Sol](https://github.com/Violet969/PLM_Sol), which gave me around 130K proteins in total. This is of course dwarfed by the millions of proteins that ESM used to train their models, but so is my wallet dwarfed when compared to theirs.

I also tried out a bunch of other exotic approaches, so wild (and utterly useless) that they don't really need any mentioning. Suffice it to say, they were bad.

But this blog post is getting pretty long and I will post the final training results as soon as it's done.

## Closing Thoughts

This was a fun exercise and I learned a TON: from using `grain` as a dataloader to sharing data between ML frameworks to predicting (albeit unsuccessfully) protein solubility. As an introduction, this was fantastic and pretty exciting!
