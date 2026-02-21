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
