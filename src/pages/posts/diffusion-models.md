
---
layout: ../../layouts/PostLayout.astro
title: Diffusion Models (DRAFT)
date: 2025-10-12
---

# Diffusion Models (DRAFT)

The overarching goal of diffusion models is to predict the noise that was added to an image at any time between $t-1$ and $t$.

![image](../../assets/Pasted%20image%2020251012115823.png)

We want to add noise to the image such that the noisy image at $X_t$ is Gaussian, i.e. ${N}(0, 1)$, which is required to create a learnable loss function and because standard Gaussians give us nice mathematical properties and, thus, are easy to work with.

Noise is added gradually, instead of one large addition of Gaussian noise. Just a bit in the beginning, then more towards the end when the image is *almost* fully Gaussian anyway. We do this, because we want to learn to undo the noise that was added between step $t$ and $t-1$.

![image](../../assets/Pasted%20image%2020251012121708.png)

The reasoning is that it's easier to learn to undo a bit of noise, rather than a lot of noise across many time steps.

### Forward Process

In order to get the noisy images to train, we need to generate them. This is naive, flawed approach:

$$
	\begin{align*}
		X_1 &= X_0 + \sqrt\beta \epsilon_1 \\
		X_2 &= X_1 + \sqrt\beta \epsilon_2 \\
		X_3 &= X_2 + \sqrt\beta \epsilon_3 \\
		\dots \\
		X_t &= X_{t-1} + \sqrt\beta \epsilon_t \\
	\end{align*}
$$
Here, $\epsilon_t$ is the Gaussian noise added at time $t$.

The $\beta$ term is added to scale down the Gaussian noise. This is required, because the input images are often normalised, usually between 0 and 1. Standard Gaussian noise can return numbers like $0.8$, which will quickly overwhelm the numbers of the input image. Therefore, the Gaussian noise needs to be scaled down.


With this current setup, we need to have the previous noisy image $X_{t-1}$ to generate the next noisy image $X_t$. But we can rewrite this to get any $X_t$ just from the starting image $X_0$. E.g.:

$$
	\begin{align*}
		X_3 &= X_2 + \sqrt\beta \epsilon_3 \\
		X_3 &= X_1 + \sqrt\beta \epsilon_2 + \sqrt\beta \epsilon_3 \\
		X_3 &= X_0 + \sqrt\beta \epsilon_1 + \sqrt\beta \epsilon_2 + \sqrt\beta \epsilon_3 \\
	\end{align*}
$$
If you have zero mean, scaled Gaussians, you can sum the **variances**, which gives you a "new" Gaussian:

$$
	\begin{align*}
	X_3 &= X_0 + 4\sqrt\beta \epsilon' \\
	X_3 &= N(X_0, 4\beta) \\
	\end{align*}
$$

But unfortunately, $X_t = N(X_0, t\beta) \neq N(0,1)$ because $X_0 \neq 0$ and $t\beta \neq 1$.

We can even see this visually if we add the noise to an image in this iterative way and compare this to what standard Gaussian noise would look like.

![image](../../assets/Pasted%20image%2020251012135210.png)

This is what the iterative approach would look like (which also takes a lot of time to run). Using the direct approach, we can get the final noisy image faster:

![image](../../assets/Pasted%20image%2020251012140010.png)

But as can be seen, the noisy image looks much different that what true Gaussian noise would look like. This further shows that our naive approach does not converge to a standard Gaussian.

A better approach is needed. The authors came up with this formula:

$$
X_t = \sqrt{1-\beta_t}X_{t-1} + \sqrt\beta_t\epsilon_t
$$
where $\beta_t$ is the scalar at time step $t$ (following a schedule e.g.). However, the issue still persists that we need to compute $X_{t-1}$ to compute $X_t$. But we'd much rather have a function that takes the initial $X_0$ and $t$ as input at outputs the correct $X_t$ directly. Let's have a look at what $X_{t-1}$ looks like.


$$
	\begin{align*}
		X_t &= \sqrt{1-\beta_t}X_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_{t-1} &= \sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1} \\
	\end{align*}
$$
If we substitute $X_{t-1}$ into the $X_t$ equation, we get


$$
	\begin{align*}
		X_t &= \sqrt{1-\beta_t}X_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_{t-1} &= \sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1} \\
		X_t &= \sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1}) + \sqrt\beta_t\epsilon_t \\

		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
	\end{align*}
$$
Now, we say that $\alpha_t = 1 - \beta_t$ to make this easier to read and have less clutter:

$$
	\begin{align*}
		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_t &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}}\epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
	\end{align*}
$$
Now we can re-use our neat trick from before where we said that variances of zero-mean Gaussians ($\epsilon_{t-1}$ and $\epsilon_t$ in our case) can be summed up. Remember, that in this equation we are working with **standard deviations** (i.e. $\sigma$) but variances are the squares of standard deviations (i.e. $\sigma^2$). This means, we need to square our standard deviations to get the variance first, i.e.:

$$
var(\sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1}) + var(\sqrt{1-\alpha_t}\epsilon_t)
$$
The variance of a Gaussian is 1, which gives us:

$$
	\begin{align*}
		& var(\sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1}) + var(\sqrt{1-\alpha_t}\epsilon_t) \\

		=& \alpha_t(1-\alpha_{t-1}) + 1-\alpha_t \\
		=& \alpha_t -\alpha_t\alpha_{t-1} + 1-\alpha_t \\
		=& 1 -\alpha_t\alpha_{t-1} \\
	\end{align*}
$$
This means that if we sum up the variances, we get a new Gaussian with mean 0 and variance $1-\alpha_t\alpha_{t-1}$. In other words:

$$
	\begin{align*}
		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_t &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}}\epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\epsilon' \\
	\end{align*}
$$
I denoted the new Gaussian noise as $\epsilon'$. From here, we can notice a pattern! We managed to describe $X_t$ as a combination from the noisy image two steps before and this has added another $\alpha$ term into the square roots. If we repeat this process until we get to $X_0$, we have this:

$$
		X_t = \sqrt{\alpha_t\alpha_{t-1}\alpha_{t-2}\dots\alpha_{1}}X_0 + \sqrt{1-\alpha_t\alpha_{t-1}\alpha_{t-2}\dots\alpha_{1}}\epsilon' \\
$$
We can simplify this a bit further by saying $\bar\alpha_t = \prod_i^t \alpha_t$, which gives us our final statement:
$$
		X_t = \sqrt{\bar\alpha_t}X_0 + \sqrt{1-\bar{\alpha_t}}\epsilon' \\
$$
And there we go! Now we can generate the final noisy image at time step $t$ just by having the initial image $X_0$. Using this new formula, we can compare the generated noisy image from what we had generated before:

![image](../../assets/Pasted%20image%2020251012140022.png)

Now this looks much closer to Gaussian noise than what we had before. It's a bit darker due to the larger value for $\beta$ that I used. However, mathematically, this does converge to a standard Gaussian

$$
		X_t = \sqrt{\bar\alpha_t}X_0 + \sqrt{1-\bar{\alpha_t}}\epsilon' \\
$$
Because as we increase $t$, we keep multiplying more and more numbers that are between 0 and 1 (because remember that $\beta$ needs to be in that range), thus making the square root smaller and smaller, which in the limit becomes 0 and $0 X_0 = 0$. Similarly, as time goes on and $\bar{\alpha_t}$ moves towards 0, the square root converges towards 1, which then just adds regular Gaussian noise via $\epsilon'$.

With this, we have the forward method covered. Now it's time to derive the loss function and the learning objective.

### Deriving the Loss Function


So far, we have described the **forward process** as a step-by-step sampling procedure. This is great for building intuition. However, to create a loss function, we need to shift from the language of single samples ($X_t$) to the language of **probability distributions**. This is important, because as we try to predict the noise added at a particular step $X_t$, we don't actually know exactly **what** noise _has to be added in general_. Think of it this way:

You have some $X_t$ and sample some noise to get $\epsilon_t$, you add that in and then get $X_{t+1}$. But this amount of noise and this specific $X_t$ are just some samples. There are many more $X_t$ and $\epsilon_t$ that could result in $X_{t+1}$, it's like a whole cloud. These specific values might have come from the edge of the cloud. If you just computed the MSE from that, you wouldn't learn the _true average_, which you would find in the center of the cloud. By thinking about this as a probability distribution, you need to find the loss between your model and the _center of the cloud_.

The formal name for the distribution defined by our single-step sampling process is **$q(X_t|X_{t-1})$**.
#### The Markov Chain Property
A crucial property of our forward process is that it's a **Markov Chain**. This simply means that the distribution for the next state, $X_t$, depends **only** on the immediately preceding state, $X_{t-1}$. It does not depend on any other previous states like $X_{t-2}$, $X_{t-3}$, or the original $X_0$.
#### The Joint Probability Distribution
We want to find the probability of an entire sequence of noisy images, $X_1, \dots, X_T$, given our starting image $X_0$. This is the joint probability distribution, **$q(X_{1:T}|X_0)$**.

Using the general **chain rule of probability**, we would write this as:
$$q(X_1, \dots, X_T | X_0) = q(X_1 | X_0) \cdot q(X_2 | X_1, X_0) \cdot \dots \cdot q(X_T | X_{T-1}, \dots, X_0)$$
This looks complicated because each step seems to depend on the entire history.

However, we can now apply our **Markov assumption**. The assumption that $X_t$ only depends on $X_{t-1}$ allows us to simplify each term:
* $q(X_2 | X_1, X_0)$ simplifies to just $q(X_2 | X_1)$.
* $q(X_T | X_{T-1}, \dots, X_0)$ simplifies to just $q(X_T | X_{T-1})$.
Applying this simplification across the entire chain gives us a much cleaner result:
$$q(X_1, \dots, X_T | X_0) = q(X_1 | X_0) \cdot q(X_2 | X_1) \cdot \dots \cdot q(X_T | X_{T-1})$$Finally, we can write this long product in a compact form using the product symbol, $\prod$: $$q(X_{1:T} | X_0) = \prod_{t=1}^{T} q(X_t | X_{t-1})$$
This final equation is the formal definition of our entire forward process.

For the reverse process, we have our model $p_\theta$. To get a clean image back, we have to compute the joint probability, i.e.:

$$
	p_\theta(X_{0:T}) = p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)
$$
From here we can start with the loss function derivation, because as stated before, we want to sample a clean image using our model. To do that, we need to minimise the negative log likelihood:

$$
L = -\log p_\theta(X_0)
$$
If we can minimise this, then we are learning the true data distribution, i.e. our dataset.

From here the first step is marginalisation using the Chain Rule of Probability:
$$
	p(A,B)=p(A∣B)p(B)
$$

$$
	\begin{align*}
		L &= -\log p_\theta(X_0) \\
		p_\theta (X_0) &=\int p_\theta (X_0∣X_{1:T})p_\theta(X_{1:T})dX_{1:T} \\
		p_\theta(X_0)&=\int p_\theta(X_{0:T})dX_{1:T} \\
	\end{align*}
$$

This loss function is numerically intractable, because we can't integrate over all possible $X$ in existence. We need to insert something into this intractable integral to make it tractable. We can use the _"divide-by-one"_ trick (I'm not sure if that's what it's called - I just like to call it that):


$$
	\begin{align*}
		p_\theta(X_0) &= \int p_\theta(X_{0:T})dX_{1:T} \\
		&= \int \frac{q(X_{1:T} | X_0)}{q(X_{1:T} | X_0)} p_\theta(X_{0:T})dX_{1:T} \\
		&= \int q(X_{1:T} | X_0)\frac{p_\theta(X_{0:T})}{q(X_{1:T} | X_0)} dX_{1:T} \\
	\end{align*}
$$

Now, this is in the form of an expectation, i.e.:

$$
	E_{p(x)}[f(x)]=∫p(x)f(x)dx
$$
Therefore, if we rewrite this into the form of an expectation, we get

$$
	\begin{align*}
		p_\theta(X_0) &= \int q(X_{1:T} | X_0)\frac{ p_\theta(X_{0:T})}{q(X_{1:T} | X_0)}   dX_{1:T} \\
		q(X_{1:T} | X_0) &\equiv q \\
		p_\theta(X_{0:T}) &\equiv p \\
		p_\theta(X_0) &= E_{q}[\frac{p}{q}]
	\end{align*}
$$
I did this definition:

$$
	\begin{align*}
		q(X_{1:T} | X_0) &\equiv q \\
		p_\theta(X_{0:T}) &\equiv p \\
	\end{align*}
$$

just so the math is a bit more concise.

Now we need to apply the logarithm:

$$
	\begin{align*}
		L &= -\log p_\theta(X_0) \\
		q(X_{1:T} | X_0) &\equiv q \\
		p_\theta(X_{0:T}) &\equiv p \\
		p_\theta(X_0) &= E_{q}[\frac{p}{q}] \\
		\log p_\theta(X_0) &= \log E_{q}[\frac{p}{q}]
	\end{align*}
$$
Because $\log$ is a concave function, we can apply Jensen's Inequality, which states:

$$
	\log(E[Y])\geq E[\log(Y)]
$$
This gives us:

$$
	\begin{align*}
		L &= -\log p_\theta(X_0) \\
		q(X_{1:T} | X_0) &\equiv q \\
		p_\theta(X_{0:T}) &\equiv p \\
		\log p_\theta(X_0) &= \log E_{q}[\frac{p}{q}] \\
		\log p_\theta(X_0) \geq& E_{q}[\log \frac{p}{q}] \\
		-\log p_\theta(X_0) \leq& -E_{q}[\log \frac{p}{q}] \\
	\end{align*}
$$

Now we have a lower bound. If we can make the expectation smaller, we will also minimise the negative log likelihood, which is exactly what we want. Thus, our new loss function to minimise is

$$
	L_n = -E_q[\log\frac{p}{q}]
$$
We can apply this log rule to get rid of the negative sign $\log(A/B)=−\log(B/A)$.


$$
	\begin{align*}
		L_n &= -E_q[\log\frac{p}{q}] \\
		&= E_q[\log\frac{q}{p}] \\
	\end{align*}
$$
Now, we can use this log rule $\log(A/B)=\log A−\log B$

$$
	\begin{align*}
		q(X_{1:T} | X_0) &= \prod_{t=1}^{T} q(X_t | X_{t-1}) \\
		q(X_{1:T} | X_0) &\equiv q \\
		p_\theta(X_{0:T}) &= p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)  \\
		p_\theta(X_{0:T}) &\equiv p \\
		L_n &= -E_q[\log\frac{p}{q}] \\
		&= E_q[\log\frac{q}{p}] \\
		&= E_q[\log{q} - \log p] \\
		L_n &= E_q[\log (\prod_{t=1}^{T} q(X_t | X_{t-1})) - \log(p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t))] \\
	\end{align*}
$$
Because $\log(A⋅B)=\log A+\log B$, we can turn the products into sums and remove the brackets of the left log part:


$$
	\begin{align*}
		L_n &= E_q[\sum_{t=1}^{T}\log q(X_t | X_{t-1}) - \log p_\theta(X_T) - \sum_{t=1}^T \log p_\theta(X_{t-1}|X_t)] \\
		&= E_q[- \log p_\theta(X_T) + \sum_{t=1}^T \log\frac{q(X_t | X_{t-1})}{p_\theta(X_{t-1}|X_t)}] \\
		&= E_q[- \log p_\theta(X_T) + \sum_{t=2}^T \log\frac{q(X_t | X_{t-1})}{p_\theta(X_{t-1}|X_t)} + \log \frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}]`
	\end{align*}
$$

You will also notice that I separated the $t=1$ step. This is required in order to apply Bayes' rule.

$$
	p(A∣B)=\frac{p(B∣A)\cdot p(A)}{p(B)}
$$
But if you instead did this:


$$
	p(A∣B,A)=\frac{p(B∣A,A)\cdot p(A|A)}{p(B|A)}
$$

You would get a so-called Dirac Delta function $p(a|a)=\delta(0)$, which is a point mass "probability distribution" which in turn isn't really a probability distribution, because there is no uncertainty. It's like saying "What card do I hold on my hand given that I hold the ace of hearts?". It's a tautology. This means you can't apply Bayes' rule here in a matter that makes sense, therefore you exclude this part.

Now we can rewrite the fraction in the sum using Bayes' Rule (and the log rule where a product becomes a sum).

$$
	\begin{align*}
					&= E_q\left[- \log p_\theta(X_T) + \sum_{t=2}^T \left(\log\frac{q(X_{t-1}|X_t, X_0)}{p_\theta(X_{t-1}|X_t)} + \log q(X_t|X_0) - \log q(X_{t-1}|X_0)\right) + \log\frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}\right]
	\end{align*}
$$


What we have now is a so called telescoping sum, which comes from this part:

$$
\sum \log q(X_t | X_0) - \log q(X_{t-1} | X_0)
$$

If you were to write out the terms you would get something like this (I will simplify this to just tuples, assume that the tuple (0,0) is equal to one, because $\log(q|q)=\log(1)=0$):

$$
	\sum^T_{t=1} = (1,0) - (0,0) + (2,0) - (1,0) + (3,0) - (2,0) + \dots
$$
The tuples being placeholders, e.g. $(1,0) \equiv \log q(X_1|X_0)$ or $(2,1) \equiv \log q(X_2|X_1)$ etc. As you can see, all the pairs cancel out except for the last one. This simplifies the sum to be this:

$$
	\begin{align*}
	    &= E_q\left[- \log p_\theta(X_T) + \sum_{t=2}^T \left(\log\frac{q(X_{t-1}|X_t, X_0)}{p_\theta(X_{t-1}|X_t)} + \log q(X_t|X_0) - \log q(X_{t-1}|X_0)\right) + \log\frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}\right] \\
		&= E_q\left[- \log p_\theta(X_T) + \sum_{t=2}^T \log\frac{q(X_{t-1}|X_t, X_0)}{p_\theta(X_{t-1}|X_t)} + \log q(X_T|X_0) - \cancel{\log q(X_1|X_0)} + \cancel{\log q(X_1|X_0)} - \log p_\theta(X_0|X_1)\right] \\
		&= E_q\left[- \log p_\theta(X_T) + \log q(X_T|X_0) + \sum_{t=2}^T \log\frac{q(X_{t-1}|X_t, X_0)}{p_\theta(X_{t-1}|X_t)} - \log p_\theta(X_0|X_1)\right]\\
		&= E_q\left[\log\frac{q(X_T|X_0)}{p(X_T)} + \sum_{t=2}^T \log\frac{q(X_{t-1}|X_t, X_0)}{p_\theta(X_{t-1}|X_t)} - \log p_\theta(X_0|X_1)\right]\\
	\end{align*}
$$
Ok, so now we have 3 terms and whenever you see a log of a quotient, you have to think "KL Divergence", which measures the _distance_ between two probability distributions, i.e.:

$$
D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$
Or for continuous probabilities:

$$
D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
$$

Or if written in the form of an expectation:

$$
D_{KL}(P || Q) = E_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right] = E_{x \sim P}[\log P(x) - \log Q(x)]
$$
This is exactly what we have in our loss term:

$$
= E_q\left[D_{KL}(q(X_T|X_0) || p(X_T)) + \sum_{t=2}^T D_{KL}(q(X_{t-1}|X_t, X_0) || p_\theta(X_{t-1}|X_t)) - \log p_\theta(X_0|X_1)\right]
$$
If we have a closer look, we can see that the left KL divergence has no trainable parameters $\theta$ in it, so we can safely ignore it, as it will become 0 as soon as we compute the gradient. As for the rightmost part, the authors of the DDPM paper have seen that, empirically, it makes no difference to leave that part it, so we can also leave that part out. This leaves us with this:

$$
= E_q\left[\sum_{t=2}^T D_{KL}(q(X_{t-1}|X_t, X_0) || p_\theta(X_{t-1}|X_t))\right]
$$
The good thing is that if your probability distributions are Gaussian, they simplify to a nice closed form. In general, for two univariate Gaussians, you have

$$
D_{KL}(P || Q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$
and for multivariate Gaussians, you have this:
$$
D_{KL}(P || Q) = \frac{1}{2}\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1}(\mu_2 - \mu_1)\right]
$$
In the above formula, $|\Sigma|$ is the determinant, $d$ is the dimensionality and $\text{tr}$ is the trace. Crucially, the authors set the **covariances** (variances) of both Gaussians to be equal, which means their determinants are also equal, which means two things:

First:
$$
	\log \frac{|\Sigma|}{|\Sigma|} = 0
$$
And second:
$$
\text{tr}(\Sigma^{-1}\Sigma) = \text{tr}(I) = d
$$
And therefore, you have this $0−d+d=0$, which means that only the means survive:

$$
D_{KL} = \frac{1}{2}(\mu_2 - \mu_1)^T \Sigma^{-1}(\mu_2 - \mu_1)
$$
Because we set the variance to a constant, $\Sigma^{-1}$ is just a constant factor, which plays no role when we minimise the loss function, so we can leave it out, which finally brings us to this part:

$$
\begin{align*}
	D_{KL}(q||p_\theta) &= \frac{1}{2}(\tilde{\mu}_t - \mu_\theta)^T(\tilde{\mu}_t - \mu_\theta) \\
	D_{KL}(q||p_\theta) &  \propto ||\tilde{\mu}_t - \mu_\theta||^2\\

L &\approx \sum_{t=2}^T E_q[||\tilde{\mu}_t(X_t, X_0) - \mu_\theta(X_t, t)||^2]
\end{align*}
$$
There is one last step, which is the reparameterisation trick. This was our forward step:

$$
X_t = \sqrt{\bar{\alpha}_t}X_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$
Which we can solve for $X_0$
$$
X_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(X_t - \sqrt{1-\bar{\alpha}_t}\epsilon)
$$
The true posterior mean has this form:
$$
\tilde{\mu}_t(X_t, X_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}X_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}X_t
$$
If we substitute $X_0$, we get:
$$
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \cdot \frac{1}{\sqrt{\bar{\alpha}_t}}(X_t - \sqrt{1-\bar{\alpha}_t}\epsilon) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}X_t
$$
Doing a bit of algebra and collecting the $X_t$ terms, we arrive at:
$$
\tilde{\mu}_t(X_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)
$$
In this form, the true mean is now expressed as a combination of $X_t$ and the true noise $\epsilon$. If we now change our neural network to output not the mean, but rather the noise directly, we get this:

$$
\mu_\theta(X_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(X_t, t)\right)
$$

If we insert this into the above loss expression, we get this:

$$
\begin{align*}
  ||\tilde{\mu}_t - \mu_\theta||^2 &= \left|\left|\frac{1}{\sqrt{\alpha_t}}\left(X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}\epsilon\right) - \frac{1}{\sqrt{\alpha_t}}\left(X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}\epsilon_\theta\right)\right|\right|^2 \\
  &= \left|\left|\frac{1}{\sqrt{\alpha_t}}\left(\cancel{X_t} - \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}\epsilon - \cancel{X_t} + \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}\epsilon_\theta\right)\right|\right|^2 \\
  &= \left|\left|\frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}(\epsilon_\theta - \epsilon)\right|\right|^2 \\
  &= \left(\frac{\beta_t}{\alpha_t(1-\bar{\alpha}t)}\right)||\epsilon_\theta - \epsilon||^2
\end{align*}
$$

Which finally brings us to

$$
L = E_{t, X_0, \epsilon}[||\epsilon - \epsilon_\theta(X_t, t)||^2]
$$
