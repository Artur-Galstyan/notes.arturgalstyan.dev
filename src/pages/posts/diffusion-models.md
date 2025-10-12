
---
layout: ../../layouts/PostLayout.astro
title: Diffusion Models (DRAFT)
date: 2025-10-12
---


The overarching goal of diffusion models is to predict the noise that was added to an image at any time between $t-1$ and $t$.

![image](../../assets/Pasted%20image%2020251012115823.png)

We want to add noise to the image such that the noisy image at $X_t$ is Gaussian, i.e. ${N}(0, 1)$, which is required to create a learnable loss function and because standard Gaussians give us nice mathematical properties and, thus, are easy to work with.

Noise is added gradually, instead of one large addition of Gaussian noise. Just a bit in the beginning, then more towards the end when the image is *almost* fully Gaussian anyway. We do this, because we want to learn to undo the noise that was added between step $t$ and $t-1$.

![image](../../assets/Pasted%20image%2020251012121708.png)

The reasoning is that it's easier to learn to undo a bit of noise, rather than a lot of noise across many time steps.

### Forward Process

In order to get the noisy images to train, we need to generate them. This is naive, flawed approach:

$$
	\begin{align}
		X_1 &= X_0 + \sqrt\beta \epsilon_1 \\
		X_2 &= X_1 + \sqrt\beta \epsilon_2 \\
		X_3 &= X_2 + \sqrt\beta \epsilon_3 \\
		\dots \\
		X_t &= X_{t-1} + \sqrt\beta \epsilon_t \\
	\end{align}
$$
Here, $\epsilon_t$ is the Gaussian noise added at time $t$.

The $\beta$ term is added to scale down the Gaussian noise. This is required, because the input images are often normalised, usually between 0 and 1. Standard Gaussian noise can return numbers like $0.8$, which will quickly overwhelm the numbers of the input image. Therefore, the Gaussian noise needs to be scaled down.


With this current setup, we need to have the previous noisy image $X_{t-1}$ to generate the next noisy image $X_t$. But we can rewrite this to get any $X_t$ just from the starting image $X_0$. E.g.:

$$
	\begin{align}
		X_3 &= X_2 + \sqrt\beta \epsilon_3 \\
		X_3 &= X_1 + \sqrt\beta \epsilon_2 + \sqrt\beta \epsilon_3 \\
		X_3 &= X_0 + \sqrt\beta \epsilon_1 + \sqrt\beta \epsilon_2 + \sqrt\beta \epsilon_3 \\
	\end{align}
$$
If you have zero mean, scaled Gaussians, you can sum the **variances**, which gives you a "new" Gaussian:

$$
	\begin{align}
	X_3 &= X_0 + 4\sqrt\beta \epsilon' \\
	X_3 &= N(X_0, 4\beta) \\
	\end{align}
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
	\begin{align}
		X_t &= \sqrt{1-\beta_t}X_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_{t-1} &= \sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1} \\
	\end{align}
$$
If we substitute $X_{t-1}$ into the $X_t$ equation, we get


$$
	\begin{align}
		X_t &= \sqrt{1-\beta_t}X_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_{t-1} &= \sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1} \\
		X_t &= \sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt\beta_{t-1}\epsilon_{t-1}) + \sqrt\beta_t\epsilon_t \\

		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
	\end{align}
$$
Now, we say that $\alpha_t = 1 - \beta_t$ to make this easier to read and have less clutter:

$$
	\begin{align}
		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_t &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}}\epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
	\end{align}
$$
Now we can re-use our neat trick from before where we said that variances of zero-mean Gaussians ($\epsilon_{t-1}$ and $\epsilon_t$ in our case) can be summed up. Remember, that in this equation we are working with **standard deviations** (i.e. $\sigma$) but variances are the squares of standard deviations (i.e. $\sigma^2$). This means, we need to square our standard deviations to get the variance first, i.e.:

$$
var(\sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1}) + var(\sqrt{1-\alpha_t}\epsilon_t)
$$
The variance of a Gaussian is 1, which gives us:

$$
	\begin{align}
		& var(\sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1}) + var(\sqrt{1-\alpha_t}\epsilon_t) \\

		=& \alpha_t(1-\alpha_{t-1}) + 1-\alpha_t \\
		=& \alpha_t -\alpha_t\alpha_{t-1} + 1-\alpha_t \\
		=& 1 -\alpha_t\alpha_{t-1} \\
	\end{align}
$$
This means that if we sum up the variances, we get a new Gaussian with mean 0 and variance $1-\alpha_t\alpha_{t-1}$. In other words:

$$
	\begin{align}
		X_t &= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}X_{t-2} + \sqrt{1-\beta_t} \sqrt\beta_{t-1}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t \\
		X_t &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}}\epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-1} + \sqrt{1-\alpha_t}\epsilon_t \\
		X_t &= \sqrt{\alpha_t\alpha_{t-1}}X_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\epsilon' \\
	\end{align}
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
