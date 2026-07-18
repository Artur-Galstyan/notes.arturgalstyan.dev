---
layout: ../../layouts/PostLayout.astro
title: What is Lab Automation? (DRAFT)
date: 2026-06-20
---

# Context

**Disclaimer: Everything here comes from my own research on the internet and is publicly available. This is just the blog of a future employee who can't wait to start and got a head start on preparing. I've not yet worked in a lab, so if you were looking for some expert opinion or insights, you've come to the wrong place!**

Sooo, it finally happened! I recently landed a new job at Roche in Basel, Switzerland and this new role I'm assuming is that of a _Lab Automation Engineer_ (LAE). This is extremely exciting and it finally lets me break free from pure fullstack development into a more challenging domain. Not to mention the fact that it's a new country I'm moving to, that is roughly a thousand kilometers away from home, which just hypes me up even more. Needless to say, I'm counting the days until I get started (which is ~3 months as of writing this).

In university I was overly eager to start learning about my classes before the semester even began and it seems as if I still carry this property even after graduating 5 years later. Currently, the issue is that although I would say that I'm a pretty good generalist builder, I'm certainly lacking domain expertise in the pharma industry. But I still have 3 months to prepare, so better get started.

Naturally, one of the first questions one might have is: what the heck even _is_ a lab automation engineer? Let's find out.

# What even is a lab??

Before we can answer what a LAE is (and does), we first have to look at what a lab even is. We all have an image in our heads: someone in a lab coat, holding a flask with some coloured liquid in it and a bunch of microscopes in the background. But perhaps we can find more grounded information. Let's ask Google. A few search results later, we find the website of Fraunhofer and specifically this image:

![Lab!](https://www.ipt.fraunhofer.de/en/offer/special-machines/laboratory-automation/jcr:content/stage/stageParsys/stage_slide/image.img.4col.jpg/1675350381521/2014-11-06-Life-Brain-090-buehne.jpg)

Now *that* is a lab if I ever saw one. Scrolling down a bit, we can see a YouTube video with another interesting photo:

![Lab!](../../assets/lab-fraunhofer_compressed.png)

By eyeballing these two images - and currently being pretty oblivious about labs in general - I would say that the first image _seems_ to be _more_ automated than the second one. But that is not really the point right now. What is interesting is what kind of devices we can spot in these images and then ask ourselves, what other devices can we usually find in labs. Each of them has a special purpose, usually doing _something_ to some liquid. 

There is no shortage of resources on the internet on this topic and you could probably fill a whole book on this, but I only have 3 months time. The 80-20 rule must be used here. We can look at [this](https://en.wikipedia.org/wiki/Category:Laboratory_equipment) Wikipedia page to get an overview but another good resource is any site that sells lab equipment (which also teaches you about what brands are out there). 

As humans, we like to categorise and put things neatly in boxes and that's what I will try to do here. I think in general you can split up lab equipment into two broad categories: (expensive as heck) electrical devices (think _oven, liquid dispencers, etc._) and single-use (-ish), generic tools (like IDK a pair of scissors or a [well](https://images.squarespace-cdn.com/content/v1/6435c4ef83064334cafe1395/8d78eb4a-1987-4184-a8cb-27c32872d932/irish-life-sciences_1.0ml-96-square-wells-low-profile-1r96-307ulp-bottom_1S96-016ULPjpg.jpeg)). The former is most interesting to us, but we shouldn't neglect the latter either. For now, we will focus on these electrical devices and if we have some time towards the end, also look at the most common instruments of the latter category.

## Frequently Used Lab Equipment

(Disclaimer: I have never worked in a lab! This list is just what I gathered from my research on the internets!)

Ok, so a lot of the work in a lab is just __moving__ liquids. Be it within the machine itself (e.g. shaking the liquid, spinning the liquid, etc.) or distributing the liquid into separate wells. 

### Pipettes

We start simple, with a relatively cheap, but probably very often used device: a pipette!

![Pipette From Gilson!](../../assets/labequipment/gilsonpipette.webp) ![Pipette From Eppendorf!](../../assets/labequipment/eppendorfpipette.webp)

By "relatively cheap" I mean in relation to the other devices we will see later in this list. Each of these bad boys can set you back between 100€ and 350€! From what I can tell, the Eppendorf brand appears a lot and seems rather expensive; I can't speak for the quality yet but so far they seem to be a bit like the "Apple" in lab equipment. Fun fact: __Eppendorf__ is the name of a district in Hamburg and as a student I used to live there. Another fun fact: just like the lab equipment, Eppendorf is one of the more beautiful (and expensive) areas to live in Hamburg.

### Shake It Up!

Shaking liquid is also common practice in a lab. I'm not 100% sure why, but the analogy I have in my head is that of juice (or oat milk, same idea). The juice particles sink towards the bottom and settle there (they are the suspended solids). Drinking unshaked juice tastes bland (you're basically just drinking fruit-water without the fruit mixed in). After shaking it, the fruity-bits and the water have mixed and your juice tastes jucier. I'm not exactly sure what the right counterparts in a biotech lab are for this analogy, let me just quickly ask Dr. Claude...

Aaaand here is the response: The above section - while true - is not actually the main reason for shaking. What I described earlier is about __homogeneity__, but the main reason is for __aeration__. AFAICT, cells need to "breathe" oxygen. Air-y water only exists on the surface at the boundary between liquid and, well, the air. If you shake the liquid, the surface gets renewed (air-y liquid from the surface mixes into the water, non-air-y water moves to the surface, absorbs a bit of oxygen, moves away, and so on) providing the cells with oxygen there and not making them suffocation to death. Otherwise, growth of the cells would mostly be concentrated near the surface and thus be limited. 

Here's an image I drew that may (or may not) help your understanding:


![Shaker Explanation!](../../assets/shaker_explanation.png)


My analogy above is also still valid: by shaking the liquid, instead of having one dense sludge of cells at the bottom, they are evenly distributed in the liquid such all cells kind of see the same particles (and nutritients) around them. 

(Shaking is also used for speeding up reactions and dissolving/resuspending)

Let's start with an orbital shaker:

![Solaris Orbital Shaker!](../../assets/labequipment/solarisorbitalshaker.png)

This unit will set you back around 4000€. Yes. You saw that right. [Here's](https://www.youtube.com/watch?v=Y1YMsl6fDOI) a video that shows you what it does. It literally just shakes the platform in a circular fashion. Why it costs so much is beyond me.

Another way to shake liquids is to use a _vortexer_. A vortexer kind of just _wiggles_ its plate around and is often used for suspension (e.g. of cells). Think back to our juice and the juice particles that have settled at the bottom. If you were to wiggle the plate on top of which it stands, then the juice particles fly into the  liquid and _suspend_ "midliquid" and fly around there. Here's one from the "Biologix" brand: 

![Vortexer!](../../assets/labequipment/vortexer.png)

(Btw. nothing here is affiliated in ANY way, I just try to find out some of the common brand names as well)

You can also often find something called a shaking incubator. The shaking part we already covered, so what's an incubator? Well it's just an enclosed container that keeps a set of parameteres constant (mostly temperature). So a shaking incubator is an orbital shaker that ensures a constant temparature. Here's one from Eppendorf:

![Vortexer!](../../assets/labequipment/eppendorfshakingincubator.png)

They can also look a bit like a microwave:

![Vortexer!](../../assets/labequipment/eppendorfshakingincubator2.png)

The latter one sets you back around 23000€, equipment so expensive, you can start to use scientific notation ($2.3e4$).

Often times, you want to grow cells to test whatever drug or compound you're developing. But cells don't just grow under any condition. They need a specific environment which stays constant, e.g. with a constant temperature or humidity. E.g. if you're growing human cells, you should make their environment close to our bodies, i.e. at a constant ~37°C. 

### Centrifuges

A centrifuge is a device that simply spins _something_ and that _something_ then experiences centrifugal forces. I'm sure most of us have been on a merry-go-round. On those, you can feel the centrifugal forces that push you out. Here's an image: 

![Merry Go Round!](../../assets/labequipment/merrygoround.png)

If you have the misfortune to ride these death traps, you will have experienced it too. They say astronauts can withstand who-knows-how-many-gs but the final NASA test is to last in these for longer than 3 minutes. 

![Death Traps!](../../assets/labequipment/kaffeetassen.png)

<div class="text-xs">Man do I hate these spinning teacups, why do they exist? Who rode those things and said "ah yes, a splendid idea"???</div>

When you have some proteins floating around in a liquid, then gravity is not enough to overcome the Brownian motion. You also don't really inspect individual proteins: in biology everything is so small that it moves towards statistics instead. You want _most_ of your proteins to be at the bottom and hopefully _most_ of those are the proteins that you were actually looking for. You also can't really "look" at a protein anyway (they are smaller than the spectrum of visible light). After you've moved most of the proteins towards the bottom of your container (which is what the centrifuges do), you can extract those, do some purification process and then perform _mass spectrometry_ so you can be _mostly certain_ about what protein you were working with. 

Here is a (chilled) centrifuge from Beckman Coulter:

![Centrifuge!](../../assets/labequipment/centrifuge.png)

I couldn't find the exact price for this (you have to request a quote), but it's probably ~6000€.

### ~Nuclear~ Bioreactor

We had already mentioned incubators. They control the environment of a closed off room, e.g. by holding a certain constant temperature. A bioreactor is related to that but instead of controlling the environment, it controls the culture. So there is a bunch of cells in your reactor and you want to control their "state". To do that, you have a few probes that read out metrics like pH, oxygen, nutrients, etc. This is something you might want to do in order to e.g. force the cells to produce a certain protein, something they might only do under the right circumstances. The bioreactor is basically what achieves and maintains that state. 

From the software point of view, a bioreactor is a constant stream of data that you have to log, understand and finally react to. 

![Bioreactor!](../../assets/labequipment/bioreactor.png)

These can get *very* expensive and for the top tier ones, scientific notation starts to make sense unironically. The one above costs around 18.000€, which is on the cheaper side from what I could find.

# What does a LAE do now?

Ok, so we already talked quite a bit about the hardware side of lab equipment, but (at least AFAICT) most of my job - in the beginning - will revolve around the software side of things, so its worth looking into that.

When it comes to lab equipment, they are pre-loaded with a lot of special software from the vendors, already able to _do_ things, perform certain actions that are generic enough for the vendor to put into the software. But what if you want to do something differently? 

Let's think of an example. Say you have a microscope from Mega Microscopes Inc. You look at some cells and then take a photo. By default, the device uploads it to the Mega Microscope Cloud. From there, you download them and pass them e.g. to some ML model, which outputs the probability of what kind of cell you're dealing with. (I know this is pretty contrived but bear with me (or actually it might not be, I haven't started yet, so something like this might actually happen)). The issue is the middle step, which is that you have to go to the cloud to download it (of course you could work around that, build a proxy, API this, post request that yada yada but that's not the point). Ideally, you'd just want the microscope to directly forward that image to your ML model, which means that you have to _change the microscopes default behaviour_.

There are 2 ways to do that: call the vendor, pay them a lot of money and let them do it. Or do it yourself. That's where a LAE comes into play. Ideally, the LAE has already talked with the scientists (communication, clear understand of what is needed to be done is very important here!) and then writes software to do what the scientists actually need. In this case, you'd have to somehow *tap into the microscrope* and do *something*.

But how?
