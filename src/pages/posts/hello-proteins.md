---
layout: ../../layouts/PostLayout.astro
title: Hello, Proteins! (DRAFT)
date: 2025-09-07
---

# Hello, Proteins! (DRAFT)

My goal for 2026 is to work at a biotech company that intersects with machine learning (if they let me of course). While language, image and video models are pretty cool (and they really are!), I think the best application for AI (until we discover AGI) is in drug discovery. And in the realm of drug discovery, proteins are **key**!

So, if my goal is to work at a biotech company such as Isomorphic Labs, Cradle or (some department of) Roche, I won't just have to grind LeetCode (that part is inevitable), I'll also have to learn more about biology in general, and proteins in particular.

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
