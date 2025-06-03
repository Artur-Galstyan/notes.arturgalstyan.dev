---
layout: ../../layouts/PostLayout.astro
title: Variational Autoencoder
date: 2025-06-01
---

# Variational Autoencoder (DRAFT)

The Variational Autoencoder (VAE) is the successor to the regular autoencoder (AE). As you probably already know, a regular AE is a neural network that compresses its input to a latent space and then reconstructs the orignal input from that latent space. In drawings, you would often find something like this:

![Standard Autoencoder](/posts/vae/autoencoder.png)

This is simple stuff, and you can easily create such a network. Here is a quick example of what an AE looks in code:

```python
class Autoencoder(eqx.Module):
    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __init__(self, dim: int, hidden_dim: int, z_dim: int, key: PRNGKeyArray):
        key, *subkeys = jax.random.split(key, 10)
        self.encoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim, hidden_dim, key=subkeys[0]),
                eqx.nn.Lambda(fn=jax.nn.relu),
                eqx.nn.Linear(hidden_dim, z_dim, key=subkeys[2]),
            ]
        )

        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(z_dim, hidden_dim, key=subkeys[2]),
                eqx.nn.Lambda(fn=jax.nn.relu),
                eqx.nn.Linear(hidden_dim, dim, key=subkeys[2]),
            ]
        )

    def __call__(self, x: Array) -> tuple[Array, Array]:
        z = self.encoder(x)
        o = self.decoder(z)
        return o, z
```

Very simple stuff. Ok, but let's say you have trained your AE on MNIST and you get get some nice reconstructions like these:

![Reconstructions](/posts/vae/reconstructions.png)


One thing you might be thinking is this: if I give my AE an image of a $1$ and I get some vector $z_1$ back and then I encode an image of a $2$ and get another vector $z_2$ back, then what does the middle point between $z_1$ and $z_2$ look like? After all, I can put either vector $z$ into my decoder and get a nice reconstructed image of my original input back. Does this mean that the middle point between the encoded vectors for the image $1$ and $2$ is an image which looks kind of like both $1$ and $2$? If it wasn't numbers but faces, can I give the AE 2 faces, take the middle point and put that through the decoder to get a completely new face back?

The unfortunate truth is: no!

But if we could somehow tidy up the latent space, then yes, we could generate new and authentic looking images. And the way we can tidy the latent space up is by using VAE.

In a VAE, we have 2 spaces: the data space $p(x)$ and the latent space $p(z)$ and we don't really have access to any of those (we only have a bunch of data points sampled from $p(x)$ but that's about it). VAE has a design decision and says that $p(z)$ is normal distributed and this will be very useful later in the loss function derivation.


Between these, we have 2 mappings (both also normal distributions) that map one space to the other which are:

$$
\begin{align*}
& p(x|z) \qquad \text{(kind of decoder)} \\
& p(z|x) \qquad \text{(kind of encoder)}
\end{align*}
$$

These mappins are like our encoder and decoder: $p(x|z)$ generates (or reconstructs) $x$ from a latent vector $z$ and vice versa.

And we don't really know those either (at least so far).

The decoder we can just learn as a supervised learning task and is by far the easiest part: we have the input $z$ and the target $x$ and all we need to do is to compare the output of the decoder against the input $x$ and we're golden.

But the encoder is a different story, because

$$
\begin{align*}
p(z|x) &= \frac{p(x|z)p(z)}{p(x)} \\
p(x) &= \int p(x|z)p(z)dz
\end{align*}
$$

which would mean we would have to integrate over the entire latent space $p(z)$, which is computationally not feasible. So, instead, we will approximate $p(z|x)$ with

$$
  q(z|x) \approx p(z|x)
$$

And we do our training correct, then $q(z|x)$ will indeed be a good approximation to the true encoder and $q(z|x)$ will also be a normal distribution. And btw. this means that it will output a $\mu$ and a $\sigma$, which we can use to sample a vector.
