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

## Data Stuff

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


def sequence_generator(fasta_path, max_seq_len, max_items):
    current_seq = []
    count = 0
    with open(pathlib.Path("data") / fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    if 0 < len(seq) <= max_seq_len:
                        yield {"sequence": seq, "length": len(seq)}
                        count += 1
                        if max_items is not None and count >= max_items:
                            return
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq and (max_items is None or count < max_items):
            seq = "".join(current_seq)
            if 0 < len(seq) <= max_seq_len:
                yield {"sequence": seq, "length": len(seq)}


def create_datasets_from_fasta(fasta_path, max_seq_len, sizes, output_dir="data"):
    for size in sizes:
        label = f"{size // 1000}k" if size is not None else "full"
        path = f"{output_dir}/uniref50_{label}"

        if pathlib.Path(path).exists():
            continue

        print(f"Creating dataset {label}...")
        ds = Dataset.from_generator(
            sequence_generator,
            gen_kwargs={
                "fasta_path": fasta_path,
                "max_seq_len": max_seq_len,
                "max_items": size,
            },
        )
        ds.save_to_disk(path)
        print(f"Saved {len(ds)} sequences to {path}")


def setup_data():
    sizes = [10_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, None]
    labels = [f"{size // 1000}k" if size is not None else "full" for size in sizes]
    create_datasets_from_fasta(
        "uniref50.fasta",
        max_seq_len=1024,
        sizes=sizes,
    )

    for size, label in zip(sizes, labels):
        dataset_name = f"data/uniref50_{label}"
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
        )
```

A couple of things are happening here. As you can see, we're splitting the UniRef50 dataset into subsets for fast iteration and prototyping. We'll be using `grain` as our dataloader and `mlflow` to track our experiments. In terms of the `max_seq_len` (which will come in just a moment) I chose 1024. Except for [Titin](https://en.wikipedia.org/wiki/Titin), which can have a length between 27,000 and 35,000 AA, I don't know of any other protein is longer than 1024. I think with 1024, we have probably 90% of the proteins covered (but I have no proof here, this is just speculation).

But now that we have our data, let's forget about this code as we will likely never touch it again. 


## Creating a Simple Baseline

We gotta start somewhere. Our baseline is going to be a simple embedding, a simple and standard transformer encoder and a fixed tokeniser. 


Let's start with the easiest: the tokeniser.

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

When working with `grain`, we can give it transformations (similar to the PyTorch dataloaders). There are two transformations that we need to apply to the sequences: the tokenisation and the masking. Here they are:

```python
def mask_sequence(
    sequence: np.ndarray,
    mask_ratio: float,
    mask_token_id: int,
    rng: np.random.Generator,
):
    aa_positions = np.where((sequence >= 1) & (sequence <= 20))[0]
    num_to_mask = int(len(aa_positions) * mask_ratio)
    mask_inds = rng.choice(aa_positions, num_to_mask, replace=False)
    sequence[mask_inds] = mask_token_id
    return sequence


class MaskMap(grain.MapTransform):
    def __init__(self, mask_ratio: float, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(seed)

    def map(self, element):
        sequence = cast(np.ndarray, element)
        masked_sequence = mask_sequence(
            sequence.copy(), self.mask_ratio, Tokenizer.MASK_ID, self.rng
        )
        return sequence, masked_sequence


class TokenizeMap(grain.MapTransform):
    max_seq_len: int

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len

    def map(self, element):
        element = cast(bytes, element)
        data = json.loads(element.decode("utf-8"))

        sequence = data["sequence"]
        length = data["length"]

        sequence = list(sequence[: min(self.max_seq_len, length)])
        sequence = (
            [Tokenizer.CLS_TOKEN]
            + sequence
            + [Tokenizer.EOS_TOKEN]
            + [Tokenizer.PAD_TOKEN for _ in range(self.max_seq_len - len(sequence))]
        )
        sequence = Tokenizer.encode(sequence)
        return np.array(sequence, dtype=np.int32)
```

## Evaluation

When you're training an encoder, it might not be as straightforward as to how to benchmark it. You usually benchmark it against how well it performs on the _downstream_ tasks (e.g. solubility prediction). But you can also directly measure the model itself, with `masked_token_accuracy` and `perplexity` being the key metrics. Here's the evaluation code for those metrics (the downstream tasks come later). 

```python
@eqx.filter_jit
def _masked_token_accuracy(_seq_logits, _mask_inds, _orig_seqs):
    pred_tokens = jnp.argmax(_seq_logits, axis=-1)
    correct = (pred_tokens == _orig_seqs) & _mask_inds
    num_masked = jnp.sum(_mask_inds)
    return jnp.where(num_masked > 0, jnp.sum(correct) / num_masked, 0.0)


@eqx.filter_jit
def _get_perplexity(_seq_logits, _mask_inds, _orig_seqs):
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        _seq_logits, _orig_seqs
    )
    masked_loss = jnp.where(_mask_inds, per_token_loss, 0.0)
    num_masked = jnp.sum(_mask_inds)
    mean_loss = jnp.where(num_masked > 0, jnp.sum(masked_loss) / num_masked, 0.0)
    return jnp.exp(mean_loss)


def evaluate(model, data_loader: grain.DataLoader, key: PRNGKeyArray):
    jitted_model = eqx.filter_jit(eqx.filter_vmap(eqx.nn.inference_mode(model)))

    masked_token_accuracies = []
    perplexities = []
    for b in data_loader:
        seqs, masked_seqs = b
        mask_inds = masked_seqs == Tokenizer.MASK_ID
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, len(seqs))
        seq_logits, embeddings = jitted_model(masked_seqs, keys)

        masked_token_accuracy = eqx.filter_vmap(_masked_token_accuracy)(
            seq_logits, mask_inds, seqs
        )
        perplexity = eqx.filter_vmap(_get_perplexity)(seq_logits, mask_inds, seqs)
        masked_token_accuracies.extend(masked_token_accuracy)
        perplexities.extend(perplexity)

    return {
        "masked_token_accuracies": jnp.mean(jnp.array(masked_token_accuracies)),
        "perplexities": jnp.mean(jnp.array(perplexities)),
    }
```

## The First Draft

First, the standard JAX boilerplate-y code:

```python
def loss_fn(model: PyTree, X, key):
    seqs, masked_seqs = X
    mask_inds = masked_seqs == Tokenizer.MASK_ID
    keys = jax.random.split(key, len(seqs))
    seq_logits, embeddings = eqx.filter_vmap(model)(masked_seqs, keys)

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(seq_logits, seqs)
    masked_loss = jnp.where(mask_inds, per_token_loss, 0.0)

    num_masked = jnp.sum(mask_inds)
    mean_loss = jnp.where(num_masked > 0, jnp.sum(masked_loss) / num_masked, 0.0)
    return mean_loss


@eqx.filter_jit
def step_fn(
    model: PyTree,
    X: tuple[Array, ...],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
):
    print("JIT") # force of habit and the easiest way to see if you reJIT the step
    value, grads = eqx.filter_value_and_grad(loss_fn)(model, X, key)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, value
```


OK, let's start with our simple baseline:

```python
class Model(eqx.Module):
    vocab_size: int
    embedding_size: int
    max_seq_len: int

    embedding: eqx.nn.Embedding
    encoder: TransformerEncoder
    rope: eqx.nn.RotaryPositionalEmbedding

    regression_head: eqx.nn.Linear

    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int,
        embedding_size: int,
        n_heads: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

        key, embedding_key, reg_key, encoder_key, pos_key = jax.random.split(key, 5)
        self.embedding = eqx.nn.Embedding(
            vocab_size, embedding_size, key=embedding_key, dtype=dtype
        )

        self.encoder = TransformerEncoder(
            embedding_size,
            n_heads=n_heads,
            num_layers=n_layers,
            key=encoder_key,
            dtype=dtype,
        )

        self.regression_head = eqx.nn.Linear(
            embedding_size,
            vocab_size,
            key=reg_key,
            dtype=dtype,
        )
        self.rope = eqx.nn.RotaryPositionalEmbedding(embedding_size // n_heads)

    def _process_heads(
        self,
        q: Float[Array, "seq_length num_heads qk_size"],
        k: Float[Array, "seq_length num_heads qk_size"],
        v: Float[Array, "seq_length num_heads vo_size"],
    ) -> tuple[
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ]:
        query_heads = eqx.filter_vmap(self.rope, in_axes=1, out_axes=1)(q)
        key_heads = eqx.filter_vmap(self.rope, in_axes=1, out_axes=1)(k)
        return query_heads, key_heads, v

    def __call__(
        self, x: Int[Array, "seq_len"], key: PRNGKeyArray
    ) -> tuple[
        Float[Array, "seq_len vocab_size"],
        Float[Array, "seq_len embedding_size"],
    ]:

        embeddings = eqx.filter_vmap(self.embedding)(x)

        pad_mask = x != Tokenizer.PAD_ID
        attn_mask = pad_mask[:, None] & pad_mask[None, :]
        attn_mask = jnp.where(attn_mask, 0.0, -jnp.inf)

        embeddings = self.encoder(
            embeddings, process_heads=self._process_heads, mask=attn_mask, key=key
        )
        seq_logits = eqx.filter_vmap(self.regression_head)(embeddings)

        return seq_logits, embeddings
```

So, wtf is going on here. First, the `TransformerEncoder` is a 1:1 copy of the `TransformerEncoder` [from PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html). Also, you might be wondering why on earth we're returning a tuple (instead of just the embeddings). This has to do with our training objective, which is MLM. 

You see, the goal is to predict what AA goes into the masked position. Therefore, we have to return some logits for that position. The embeddings are directly used for that objective. In a way, we're already optimising for a downstream task (the MLM). If we have rich embeddings, the regression head that predicts the AA will have an easy time. So this also means that our loss function (which - again - computes the loss between the token we predicted at the mask position vs. what was actually there), needs the sequence logits. 

The `attn_mask` is there to make sure we don't accidentally train on the padding tokens (which we added to fill the `max_seq_len`).

Other than that, the process is straightforward: tokenised sequence -> embeddings -> encoder -> return.

Let's do a first training run, using these hyperparameters:
```python
max_seq_len = 256
mask_ratio = 0.25
batch_size = 256
embedding_size = 768
n_heads = 8
n_layers = 8
```

This is the full training code:

```python
def make_eval_loader(data_source, transformations):
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=False,
        seed=0,
    )
    return grain.DataLoader(
        data_source=data_source,
        operations=transformations,
        sampler=sampler,
        worker_count=0,
    )


def main():
    setup_data()
    setup_mlflow()
    # dataset_name = "data/uniref50_1000k"
    dataset_name = "data/uniref50_10k"
    train_source = HFDataSource(f"{dataset_name}/train")
    val_source = HFDataSource(f"{dataset_name}/val")

    mesh_size = (len(jax.devices()),)
    # mesh_size = (1,)
    mesh = jax.make_mesh(
        mesh_size, axis_names=("batch",), axis_types=(js.AxisType.Auto,)
    )
    data_sharding = js.NamedSharding(
        mesh,
        js.PartitionSpec(
            "batch",
        ),
    )
    model_sharding = js.NamedSharding(mesh, js.PartitionSpec())

    n_epochs = 50

    max_seq_len = 256
    mask_ratio = 0.25
    batch_size = 256
    embedding_size = 768
    n_heads = 8
    n_layers = 8

    transformations = [
        TokenizeMap(max_seq_len),
        MaskMap(mask_ratio),
        grain.Batch(batch_size=batch_size),
    ]

    train_index_sampler = grain.IndexSampler(
        num_records=len(train_source),
        num_epochs=n_epochs,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=0,
    )
    train_data_loader = grain.DataLoader(
        data_source=train_source,
        operations=transformations,
        sampler=train_index_sampler,
        worker_count=0,
    )

    model = Model(
        max_seq_len,
        Tokenizer.VOCAB_SIZE,
        embedding_size,
        n_heads=n_heads,
        n_layers=n_layers,
        key=jax.random.key(100),
        dtype=jnp.float32,
    )

    batches_per_epoch = len(train_source) // batch_size

    decay_steps = batches_per_epoch * n_epochs
    warmup_steps = min(1000, decay_steps // 10)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    current_epoch = 0
    avg_loss = 0
    key = jax.random.key(42)

    model_size = param_summary(model, True)

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_seq_len", max_seq_len)
        mlflow.log_param("embedding_size", embedding_size)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("model_size", model_size)
        mlflow.log_param("dataset_name", dataset_name)

        for i, batch in tqdm(enumerate(train_data_loader)):
            epoch = i // batches_per_epoch
            X = batch
            X = eqx.filter_shard(X, data_sharding)
            key, subkey = jax.random.split(key)
            model, opt_state, loss = step_fn(model, X, optimizer, opt_state, subkey)

            avg_loss += loss.item()
            if epoch > current_epoch:
                current_epoch = epoch
                valid_data_loader = make_eval_loader(val_source, transformations)
                key, subkey = jax.random.split(key)
                evals = evaluate(model, valid_data_loader, subkey)
                avg_loss /= batches_per_epoch
                for k in evals:
                    evals[k] = evals[k].item()
                print(f"Epoch {epoch}: {evals};\navg_loss={avg_loss}")
                current_lr = schedule(i).item()
                mlflow.log_metrics(
                    {**evals, "avg_loss": avg_loss, "learning_rate": current_lr},
                    step=epoch,
                )
                avg_loss = 0
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, "model.eqx")
                    eqx.tree_serialise_leaves(model_path, model)
                    mlflow.log_artifact(model_path, f"model-{epoch}")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.eqx")
            eqx.tree_serialise_leaves(model_path, model)
            mlflow.log_artifact(model_path, "model")


if __name__ == "__main__":
    main()
```

This model has 22M parameters:

| Layer | Shape | Params |
| :--- | :--- | ---: |
| `.embedding.weight` | `(25, 768)` | 19,200 |
<details>

|  |  |  |
| :--- | :--- | ---: |
| `.encoder.layers0.self_attn.query_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers0.self_attn.query_proj.bias` | `(768,)` | 768 |
| `.encoder.layers0.self_attn.key_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers0.self_attn.key_proj.bias` | `(768,)` | 768 |
| `.encoder.layers0.self_attn.value_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers0.self_attn.value_proj.bias` | `(768,)` | 768 |
| `.encoder.layers0.self_attn.output_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers0.self_attn.output_proj.bias` | `(768,)` | 768 |
| `.encoder.layers0.linear1.weight` | `(2048, 768)` | 1,572,864 |
| `.encoder.layers0.linear1.bias` | `(2048,)` | 2,048 |
| `.encoder.layers0.linear2.weight` | `(768, 2048)` | 1,572,864 |
| `.encoder.layers0.linear2.bias` | `(768,)` | 768 |
| `.encoder.layers0.norm1.weight` | `(768,)` | 768 |
| `.encoder.layers0.norm1.bias` | `(768,)` | 768 |
| `.encoder.layers0.norm2.weight` | `(768,)` | 768 |
| `.encoder.layers0.norm2.bias` | `(768,)` | 768 |
| `.encoder.layers1.self_attn.query_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers1.self_attn.query_proj.bias` | `(768,)` | 768 |
| `.encoder.layers1.self_attn.key_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers1.self_attn.key_proj.bias` | `(768,)` | 768 |
| `.encoder.layers1.self_attn.value_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers1.self_attn.value_proj.bias` | `(768,)` | 768 |
| `.encoder.layers1.self_attn.output_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers1.self_attn.output_proj.bias` | `(768,)` | 768 |
| `.encoder.layers1.linear1.weight` | `(2048, 768)` | 1,572,864 |
| `.encoder.layers1.linear1.bias` | `(2048,)` | 2,048 |
| `.encoder.layers1.linear2.weight` | `(768, 2048)` | 1,572,864 |
| `.encoder.layers1.linear2.bias` | `(768,)` | 768 |
| `.encoder.layers1.norm1.weight` | `(768,)` | 768 |
| `.encoder.layers1.norm1.bias` | `(768,)` | 768 |
| `.encoder.layers1.norm2.weight` | `(768,)` | 768 |
| `.encoder.layers1.norm2.bias` | `(768,)` | 768 |
| `.encoder.layers2.self_attn.query_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers2.self_attn.query_proj.bias` | `(768,)` | 768 |
| `.encoder.layers2.self_attn.key_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers2.self_attn.key_proj.bias` | `(768,)` | 768 |
| `.encoder.layers2.self_attn.value_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers2.self_attn.value_proj.bias` | `(768,)` | 768 |
| `.encoder.layers2.self_attn.output_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers2.self_attn.output_proj.bias` | `(768,)` | 768 |
| `.encoder.layers2.linear1.weight` | `(2048, 768)` | 1,572,864 |
| `.encoder.layers2.linear1.bias` | `(2048,)` | 2,048 |
| `.encoder.layers2.linear2.weight` | `(768, 2048)` | 1,572,864 |
| `.encoder.layers2.linear2.bias` | `(768,)` | 768 |
| `.encoder.layers2.norm1.weight` | `(768,)` | 768 |
| `.encoder.layers2.norm1.bias` | `(768,)` | 768 |
| `.encoder.layers2.norm2.weight` | `(768,)` | 768 |
| `.encoder.layers2.norm2.bias` | `(768,)` | 768 |
| `.encoder.layers3.self_attn.query_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers3.self_attn.query_proj.bias` | `(768,)` | 768 |
| `.encoder.layers3.self_attn.key_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers3.self_attn.key_proj.bias` | `(768,)` | 768 |
| `.encoder.layers3.self_attn.value_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers3.self_attn.value_proj.bias` | `(768,)` | 768 |
| `.encoder.layers3.self_attn.output_proj.weight` | `(768, 768)` | 589,824 |
| `.encoder.layers3.self_attn.output_proj.bias` | `(768,)` | 768 |
| `.encoder.layers3.linear1.weight` | `(2048, 768)` | 1,572,864 |
| `.encoder.layers3.linear1.bias` | `(2048,)` | 2,048 |
| `.encoder.layers3.linear2.weight` | `(768, 2048)` | 1,572,864 |
| `.encoder.layers3.linear2.bias` | `(768,)` | 768 |
| `.encoder.layers3.norm1.weight` | `(768,)` | 768 |
| `.encoder.layers3.norm1.bias` | `(768,)` | 768 |
| `.encoder.layers3.norm2.weight` | `(768,)` | 768 |
| `.encoder.layers3.norm2.bias` | `(768,)` | 768 |
| `.encoder.norm.weight` | `(768,)` | 768 |
| `.encoder.norm.bias` | `(768,)` | 768 |

  <summary>
    Other layers...
  </summary>
</details>

|  |  |  |
| :--- | :--- | ---: |
| `.regression_head.weight` | `(25, 768)` | 19,200 |
| `.regression_head.bias` | `(25,)` | 25 |
| **Total** | | **22,095,897** |

We're training this on the smallest data subset (the 10k version) for now to get some data quickly.
