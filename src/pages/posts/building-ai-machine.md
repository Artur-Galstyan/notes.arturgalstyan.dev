---
layout: ../../layouts/PostLayout.astro
title: Building an AI PC (DRAFT)
date: 2025-12-14
---

If you want to develop ML models, you have a few options, depending on the kind of models that you want to develop. Do you want to train LLMs (emphasis on the first "L")? Yes? Ok, are you incredibly rich ($>20m$ net worth)? Yes? Then you can go to NVIDIA and tell them to make you a 8x H200 server and send it to your home. You'll need to call your electrician though in order to turn the thing on. 

You aren't rich and still want to work on cutting edge, very large LLMs? I'd say, give up. This isn't a game meant to be played by the average mortal. You won't be able to afford the GPU costs to significantly move the needle in the LLM space. 

Smaller LLMs or models in general that are $<500m$ params? Now we're talking. Those are the kinds of models that you can train in a limited amount of time to a meaningful amount of performance.

For me, I don't care too much about very large LLMs AND I don't have much money. So I have really only 3 options for the GPUs:

- NVIDIA RTX 5090 (~2800€)
- NVIDIA RTX Pro 4000 (~1400€)
- NVIDIA RTX Pro 4500 (~2800€)

Out of these, the 5090 is king. It has the highest VRAM (32 GB), the most AI TOPS and (most unfortunately) the highest power draw as well (575W - ouch!) For training, I'd say this is the best candidate. The RTX Pro 4000 and 4500 are not bad and very power efficient (important for large datacenters) but I'd say are most suitable for inference tasks. So, the 5090 it is!

But before you go ahead and buy 2 5090s, make sure that your home **can** even run such a computer. Having two 5090s (and also the rest of the PC) means that you will need to have a PSU with at least 2200W, ideally 2500W (the [hela](https://www.amazon.de/Silverstone-Cybenetics-vollst%C3%A4ndig-ATX-Netzteil-SST-HA2500R-PM/dp/B0FF9X8CM7) is a good PSU for this - non affiliate link by the way). If you are in the US, your wall might not even allow you to pull that much energy (granted, you wan't always reach continious 2500W, but realistically between 1500W and 2000W). For us in Europe our walls provide 240V and especially in Germany, you can comfortably pull around 3000W. If you need more, then look behind your oven, you could find a high voltage adapter there.

**DO NOT TOUCH THAT UNLESS YOU ARE A PROFESSIONAL! INTERACTING WITH MAINS VOLTAGE CAN KILL YOU!**

Ok but let's assume your wall provides enough energy, what else do you need?

For one, you'll need a fitting CPU such that it doesn't bottleneck your GPUs. But not any CPU will work. You'd think that if you just bought the latest i9, surely that would be enough but you're wrong. The important property to look out for is PCIE lanes.
