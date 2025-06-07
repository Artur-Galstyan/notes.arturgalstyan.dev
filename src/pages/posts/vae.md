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

But if we could somehow tidy up the latent space, then yes, we could generate new and authentic looking images. And the way we can tidy the latent space up is by using VAE. VAEs solve this by forcing the latent space to follow a specific distribution (Gaussian), which creates a smooth, organized latent space where interpolation works!

In a VAE, we have 2 spaces: the data space $p(x)$ and the latent space $p(z)$ and we don't really have access to any of those (we only have a bunch of data points sampled from $p(x)$ but that's about it). VAE has a design decision and says that $p(z)$ is normal distributed and this will be very useful later in the loss function derivation.


Between these, we have 2 mappings (both also normal distributions) that map one space to the other which are:

$$
\begin{align*}
& p(x|z) \qquad \text{(kind of decoder)} \\
& p(z|x) \qquad \text{(kind of encoder)}
\end{align*}
$$

These mappings are like our encoder and decoder: $p(x|z)$ generates (or reconstructs) $x$ from a latent vector $z$ and vice versa.

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

And we do our training correctly, then $q(z|x)$ will indeed be a good approximation to the true encoder and $q(z|x)$ will also be a normal distribution. And btw. this means that it will output a $\mu$ and a $\sigma$, which we can use to sample the vector $z$.

## Deriving the Loss Function

Here's the goal: we want to maximise $log p(x)$ for each $x$ in our dataset and if you're asking why, then you're in good company, because that's not immediately obvious. Remember how I said we have 2 distributions and that one of them is $p(x)$ - the data distribution? What $log p(x)$ tells us is the probability that we could sample $x$ from the distribution. But all our data points $x$ ARE from the data distribution, so the probability is $1.0$, because we don't have any datapoints outside of our dataset. So that is our starting point:

$$
  \text{maximise} \qquad \log p(x)
$$

The first thing we can say is this:

$$
  \log p(x) = \log \int p(x|z)p(z) dz
$$

And that means _marginalising_ out $z$. To better understand this, imagine you had 2 die (an $x$-dice and a $z$-dice) and their probabilities are skewed such that higher numbers have higher probabilities. A probability matrix would look like this:

![Probability Matrix](/posts/vae/matrix.png)

The redder areas indicate a higher probability. If you were interested in the probability that $p(x=5)$, then, to calculate that, you need to compute the sum of all the outcomes where $p(x=5, z=any)$ (that's the one I highlighted in the image), so, in other words, it's:

$$
  p(x=5) = \sum_{i=1}^6 p(x=5, z=i)
$$

This process is called _marginalization_. It's essentially the same as $\log p(x) = \log \int p(x|z) dz$, except of course there we are dealing with continious values (and the log is there for numerical stability, but doesn't change the probabilities underneath).

We use the "_multiply by one_" trick to introduce a new term:
$$
  \begin{align*}
\log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
&= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
&= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right)
 \end{align*}
$$

Pretty neat, now we introduced our approximation. We can rearrange some stuff to get this:


$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right)
 \end{align*}
$$

The next trick is a bit unintuitive, but bear with me. Let's say you have two functions:

$$
  f(z) = z
$$

and another function which generates the $z$ randomly

$$
  Q(z)
$$

Think back to the skewed die from earlier. If $Q(z)$ is the random outcome generator for one of those die, then it will return higher values for $z$ with greater probability than lower ones. So what is the expected value for the function $f(z)$ in this case? It is defined as:

$$
  \mathbb{E}_{Q(z)} f(z) = \int Q(z) f(z) dz
$$

Or spoken in plain English: the expected value for the function $f(z)$ is the probability to sample a $z$ times the value of that $z$. For our die example, we could say that we have a 10% change to roll a 1 and a 90% chance to roll a 6 and nothing else. In this case, the expected value for $f(z)$ is

$$
  \begin{align*}
    f(z) &= z \\
    f(1) &= 1 \\
    f(6) &= 6 \\
    Q(1) &= 0.1 \\
    Q(6) &= 0.9 \\

    \mathbb{E} &= Q(1) * f(1) + Q(6) * f(6) \\
     &= 0.1 * 1 + 0.9 * 6 \\
     &= 5.5
  \end{align*}
$$

We have the same setting in our derivation:

$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right)
 \end{align*}
$$

Where $q(z|x)$ is the probability to sample $z$ (this is akin to the Q(z) from the definition earlier) and $\frac{p(x,z)}{q(z|x)}$ is the value function (the f(z) in the example). With that, we can rewrite it like so:


$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right) \\
   &= \log \mathbb{E}_{q(z|x)} \left( \frac{p(x,z)}{q(z|x)} \right)
 \end{align*}
$$

The next trick we can use is the Jensen inequality, which states:

$$
f(E(y)) \ge E(f(y))
$$

if $f$ is a concave function and since $\log$ is a concave function, we can say

$$
\log(E(y)) \ge E(\log(y))
$$

For our derivation, we can now write:

$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right) \\
   &= \log \mathbb{E}_{q(z|x)} \left( \frac{p(x,z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log \frac{p(x,z)}{q(z|x)} \right)
 \end{align*}
$$

This is great, because now if we can - somehow - increase $\mathbb{E}_{q(z|x)} \left( \log \frac{p(x,z)}{q(z|x)} \right)$, then it will automatically raise the bar for $\log p(x)$.

Because $p(x,z) = p(x|z)p(z)$, we can write and rearrange the terms like so:

$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right) \\
   &= \log \mathbb{E}_{q(z|x)} \left( \frac{p(x,z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log \frac{p(x,z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log \frac{p(x|z)p(z)}{q(z|x)} \right)
 \end{align*}
$$

The laws of the logs tell us:

$$
  \begin{align*}
    \log(xy) &= \log x + \log y \\
    \log(x/y) &= \log x - \log y
  \end{align*}
$$

And because of that, we can rewrite the term as:

$$
  \begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x,z) dz \right) \\
   &= \log \left( \int q(z|x) \frac{p(x,z)}{q(z|x)}  dz \right) \\
   &= \log \mathbb{E}_{q(z|x)} \left( \frac{p(x,z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log \frac{p(x,z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log \frac{p(x|z)p(z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \log p(x|z) + \log \frac{p(z)}{q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)} \left( \textcolor{blue}{\log p(x|z)} + \textcolor{green}{\log p(z) - \log q(z|x)} \right) \\
   &\ge \mathbb{E}_{q(z|x)}(\log p(x|z)) + \mathbb{E}_{q(z|x)}(\log p(z) - \log q(z|x)) \\
 \end{align*}
$$

The blue part trains the decoder, while the green part trains the encoder. Furthermore, the blue part will simplify to the MSE, while the green part is the exact definition of the *negative* KL divergence. Let's start with the decoder part, because that's a bit easier.

I said earlier that the encoder outputs a $\mu$ and a $\sigma$ which we use to sample the latent vector $z$. The decoder technically also outputs both of these, but in practice, we set $\sigma$ to a constant and use $\mu$ directly. Because the decoder is a normal distribution, we can write this:

$$
\begin{align*}
p(x|z) &= \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|x - x_{rec}(z)\|^2}{2\sigma^2}\right) \\
\log p(x|z) &= \log\left( \frac{1}{(2\pi\sigma^2)^{D/2}} \right) - \frac{\|x - x_{rec}(z)\|^2}{2\sigma^2}
\end{align*}
$$

When it comes to optimization, everything that is a constant, we don't care about. This means that the only thing that does remain and is NOT constant is:

$$
{(x - x_{rec}(z))}^2
$$

Which is the mean squared error (and $x_{rec}$ is the output of our decoder). Now, let's have a look at the encoder:

$$
\mathbb{E}_{q(z|x)}(\log p(z) - \log q(z|x))
$$

Which is precisely the definition for the KL divergence, and because $p(z)$ is a normal distribution, the KL divergence simplifies to a closed form:

$$
\begin{align*}
D_{KL}(q(z|x) \,||\, p(z))
&= \log\frac{\sigma_z}{\sigma_e} + \frac{\sigma_e^2 + (\mu_e - \mu_z)^2}{2\sigma_z^2} - \frac{1}{2} \\
&= \log\frac{1}{\sigma_e} + \frac{\sigma_e^2 + (\mu_e - 0)^2}{2 \cdot 1^2} - \frac{1}{2} \\
&= -\log\sigma_e + \frac{\sigma_e^2 + \mu_e^2}{2} - \frac{1}{2} \\
&= -\frac{1}{2}\log(\sigma_e^2) + \frac{\sigma_e^2 + \mu_e^2 - 1}{2} \\
&= \frac{1}{2} \left( \mu_e^2 + \sigma_e^2 - \log(\sigma_e^2) - 1 \right) \\
&= -\frac{1}{2} \left( 1 + \log(\sigma_e^2) - \mu_e^2 - \sigma_e^2 \right)
\end{align*}
$$

$\mu_z$ and $\sigma_z$ come from $p(z)$ and because $p(z)$ is a Gaussian, those are $0$ and $1$ respectively and $\mu_e$ and $\sigma_e$ come from $q(z|x)$ (i.e. the encoder approximation).

So, if we put everything together, we get:

$$
\begin{align*}
  \log p(x) &= \log \left( \int p(x|z)p(z) dz \right) \\
   &= \log \left( \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \right) \\
&= \log \int \frac{q(z|x)}{q(z|x)} p(x|z)p(z) dz \\
&= \log \int q(z|x) \frac{p(x|z)p(z)}{q(z|x)} dz \\
&= \log \mathbb{E}_{q(z|x)} \left[ \frac{p(x|z)p(z)}{q(z|x)} \right] \\
&\geq \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x|z)p(z)}{q(z|x)} \right] \quad \text{(Jensen's inequality)} \\
&= \mathbb{E}_{q(z|x)} \left[ \log p(x|z) + \log p(z) - \log q(z|x) \right] \\
&= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}[\log p(z) - \log q(z|x)] \\
&= \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \,||\, p(z)) \quad \text{(this is the ELBO)}\\
\log p(x|z) &= -\frac{1}{2}||x - x_{rec}(z)||^2 - \frac{d}{2}\log(2\pi) \\
D_{KL}(q(z|x) \,||\, p(z)) &= \frac{1}{2}\sum(\mu_e^2 + \sigma_e^2 - \log \sigma_e^2 - 1) \\
\text{Loss} &= -\text{ELBO} \\
&= -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \,||\, p(z)) \\
&= -\mathbb{E}_{q(z|x)}\left[-\frac{1}{2}||x - x_{rec}(z)||^2 - \frac{d}{2}\log(2\pi)\right] + D_{KL}(q(z|x) \,||\, p(z)) \\
&= \mathbb{E}_{q(z|x)}\left[\frac{1}{2}||x - x_{rec}(z)||^2 + \frac{d}{2}\log(2\pi)\right] + D_{KL}(q(z|x) \,||\, p(z)) \\
&= \frac{1}{2}\mathbb{E}_{q(z|x)}[||x - x_{rec}(z)||^2] + \frac{1}{2}\sum(\mu_e^2 + \sigma_e^2 - \log \sigma_e^2 - 1) + \text{const} \\
\mathcal{L} &= \mathbb{E}_{q(z|x)}[||x - x_{rec}(z)||^2] + \frac{1}{2}\sum(\mu_e^2 + \sigma_e^2 - \log \sigma_e^2 - 1)
\end{align*}
$$

The $+\text{const}$ part refers to this part $\frac{d}{2}\log(2\pi)$ which is just some constant and those don't matter when we minimise in our ML libraries and the same goes for scalars, which is why we dropped the $1/2$ too.
