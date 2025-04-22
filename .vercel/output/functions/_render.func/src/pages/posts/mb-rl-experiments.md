---
layout: ../../layouts/PostLayout.astro
title: Model-Based Reinforcement Learning Experiments
date: 2025-04-22
---

# Model-Based Reinforcement Learning

Model-Based Reinforcement Learning (MBRL) is one of my favourite areas in machine learning in general. IMO, MBRL is likely going to be integral to finding AGI one day, because I believe that AGI (and intelligence in general) is simply a matter of _goal oriented next-state prediction_. I will leave this simply as an opinion and showcase my arguments for this at another time, because in this post, it's all about coding MBRL.

Btw. this is __not__ a tutorial, but rather a workthrough of MBRL and some of the issues encountered along the way.

Let's start with the basics first. The goal of MBRL is to learn a model of the environment. The idea being that if you had a _perfect_ model of the environment, then you could use that to plan a sequence of actions to get to your desired goal state. The difficulty is now to train that model, which should take in the current state $s$ and the performed action $a$ as input and predict the next state $s'$. For this, we will just use a simple MLP (btw. we are training on CartPole for now - gotta start somewhere, right?):

```python
class DynamicsModel(eqx.Module):
    mlp: eqx.nn.MLP

    n_dims: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, n_dims: int, n_actions: int, key: PRNGKeyArray):
        self.mlp = eqx.nn.MLP(
            in_size=n_dims + n_actions,
            out_size=n_dims,
            width_size=128,
            depth=3,
            key=key,
        )
        self.n_dims = n_dims
        self.n_actions = n_actions

    def __call__(
        self, x: Float[Array, " n_dims+n_actions"]
    ) -> Float[Array, "n_dims"]:
        state = x[: self.n_dims]
        delta = self.mlp(x)
        return state + delta
```

Admittedly, this is not the purest MLP, because we are using residual connections, i.e. we are trying to learn the delta between the current state and the next state. The argument is that usually there is not such a drastic change between states, thus being easier to learn the delta than the whole thing.

Ok, so now we have a dynamics prediction model (we have no idea yet if it works or not though). One thing we will need for sure is __data__! One of the advantages of MBRL is that all data is good data, in the sense that you have no reason to throw out any data, since all the data you collect comes from the environment i.e. the ground truth.

```python
def collect_data(
    env: gym.Env,
    n_episodes: int,
    dynamics_model,
    reward_model,
    epsilon: float = 0.1,
    n_horizon: int = 10,
) -> list[tuple[Array, Array, Array, Array]]:
    data = []
    for i in range(n_episodes):
        state, _ = env.reset()

        terminated, truncated = False, False
        while not terminated and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(????)
            next_state, reward, terminated, truncated, _ = env.step(action)
            data.append((state, action, reward, next_state))
    return data
```

The above function is simple: just create `n_episode` worth of data using epsilon-greedy exploration, i.e. 10 % random exploration (in this example). But what about the `else` part?

In the `else` branch, we have to specify _how_ we should - in general - choose actions based on our model. I alluded to this earlier, namely that we need to plan ahead. Following the [talk from Sergey Levine](https://www.youtube.com/watch?v=VxyhEK4yW5g), you could use the cross-entropy method (CEM), which is a gradient-free planning method, although there are certainly other methods as well, such as Monte-Carlo Tree Search (MCTS). We will use CEM for now and later use MCTS to see the difference.

## Excursion: Cross-Entropy Method
