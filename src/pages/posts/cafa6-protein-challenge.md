---
layout: ../../layouts/PostLayout.astro
title: Kaggle Competition CAFA 6 - Protein Function Prediction (DRAFT)
date: 2025-12-22
---

One afternoon, I was very bored and browsed through Kaggle competitions, unsure as to why because I don't actually participate in Kaggle competitions. But there I spotted the CAFA 6 challenge in which you have to predict functions of proteins. 

I'm trying to break into biotech as a SWE, but needless to say, finding such a job is extremely difficult. And here I had an idea: wouldn't it be sweet to place high in such a challenge, which a) teaches you a lot about protein function prediction (PFP - I don't know if that's the official abbreviation, but it is for this blog post) and b) gives companies an incentive to hire you (because who would want to pass on someone with a gold medal in PFP).

A gold medal is currently just a pipe dream, so let's aim for something more realistic and then adjust our expectations as we go. For now, the goal is top 200!

## What is this Challenge?

This is the 6th iteration of the CAFA (Critical Assessment of Functional Annotation) challenge. The goal: given some protein sequence, determine the function of the given protein. Actually, quite simple.

Ok, but "as what" do we predict the function? Is it a string, a number or what is it? Well, in protein-land, to describe the function of a protein, you can use _Gene Ontology_ (GO) terms. They describe what a protein does.

## What Does the Data Look Like?

This kaggle challenge provides you with multiple files, one of those is called `train_sequences.fasta`. Such `.fasta` files are AFAICT the standard format for proteins. Here's an example from the dataset:

``` 
>sp|A1A519|F170A_HUMAN Protein FAM170A OS=Homo sapiens OX=9606 GN=FAM170A PE=1 SV=1
MKRRQKRKHLENEESQETAEKGGGMSKSQEDALQPGSTRVAKGWSQGVGEVTSTSEYCSC
VSSSRKLIHSGIQRIHRDSPQPQSPLAQVQERGETPPRSQHVSLSSYSSYKTCVSSLCVN
KEERGMKIYYMQVQMNKGVAVSWETEETLESLEKQPRMEEVTLSEVVRVGTPPSDVSTRN
LLSDSEPSGEEKEHEERTESDSLPGSPTVEDTPRAKTPDWLVTMENGFRCMACCRVFTTM
EALQEHVQFGIREGFSCHVFHLTMAQLTGNMESESTQDEQEEENGNEKEEEEKPEAKEEE
GQPTEEDLGLRRSWSQCPGCVFHSPKDRNS
```

We get quite a bit of information here, but let's focus on two right now. First, the protein id is the second value in the first line, i.e. `A1A519` - we will use this later to look up its GO terms. Then, the most important piece of information, is the amino acid sequence `MKRRQKRKH...`. Ok, so what does this particular protein do? Let's have a look in the `train_terms.tsv` file (i.e. our labels, what we want to predict). Searching for the GO terms for this protein (`A1A519`), we find these:

```tsv 
EntryID	term	aspect
A1A519	GO:0005634	C
A1A519	GO:0045893	P
A1A519	GO:0006366	P
```

The aspect is the subontology, i.e. Molecular Function (MF), Biological Process (BP), and Cellular Component (CC), specifically:

- P → Biological Process (BP)
- F → Molecular Function (MF)
- C → Cellular Component (CC)

Let's look what these GO terms stand for. 

```
[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
def: "A membrane-bounded organelle of eukaryotic cells in which chromosomes are housed and replicated. In most cells, the nucleus contains all of the cell's chromosomes except the organellar chromosomes, and is the site of RNA synthesis and processing. In some species, or in specialized cell types, RNA metabolism or DNA replication may be absent." [GOC:go_curators]
[...]
subset: goslim_yeast
synonym: "cell nucleus" EXACT []
synonym: "horsetail nucleus" NARROW [GOC:al, GOC:mah, GOC:vw, PMID:15030757]
xref: NIF_Subcellular:sao1702920020
xref: Wikipedia:Cell_nucleus
is_a: GO:0043231 ! intracellular membrane-bounded organelle
```

```
[Term]
id: GO:0045893
name: positive regulation of DNA-templated transcription
namespace: biological_process
alt_id: GO:0043193
alt_id: GO:0045941
alt_id: GO:0061020
def: "Any process that activates or increases the frequency, rate or extent of cellular DNA-templated transcription." [GOC:go_curators, GOC:txnOH]
synonym: "activation of gene-specific transcription" RELATED []
synonym: "activation of transcription, DNA-dependent" NARROW []
[...]
synonym: "upregulation of transcription, DNA-dependent" EXACT []
is_a: GO:0006355 ! regulation of DNA-templated transcription
is_a: GO:1902680 ! positive regulation of RNA biosynthetic process
relationship: positively_regulates GO:0006351 ! DNA-templated transcription
```

```
[Term]
id: GO:0006366
name: transcription by RNA polymerase II
namespace: biological_process
alt_id: GO:0032568
alt_id: GO:0032569
def: "The synthesis of RNA from a DNA template by RNA polymerase II (RNAP II), originating at an RNA polymerase II promoter. Includes transcription of messenger RNA (mRNA) and certain small nuclear RNAs (snRNAs)." [GOC:jl, GOC:txnOH, ISBN:0321000382]
subset: goslim_yeast
synonym: "gene-specific transcription from RNA polymerase II promoter" RELATED []
synonym: "general transcription from RNA polymerase II promoter" RELATED []
[...]
xref: Reactome:R-HSA-73857 "RNA Polymerase II Transcription"
is_a: GO:0006351 ! DNA-templated transcription
```

Ok, so this protein is involved in the transcription (by RNA polymerase and positive regulation of DNA-templated transcription) in the cell nucleus. Looking at the second GO term (GO: 0045893), it basically just says "anything that has to do with transcription". But this isn't really helpful. I'm not a biologist and even I could have guessed this.

In a similar vein, if we predict this label - since it appliest to almost all proteins, we shouldn't praise our model too much for this. It's like getting an image of a car and your model predicts "object", then you get another image of a crane and you predict "object" again. Technically correct, practically useless. It's the same with PFP. 

The challenge has a way to deal with this and force your model to predict much more specific terms. You get an extra file called `IV.tsv` (which stands for Information Accretion). In there, you can see how much predicting a certain GO term is "worth". For instance, our GO terms have:

```tsv 
term	weight
GO:0045893	0.0
GO:0006366	0.5273799211477056
GO:0005634	0.9534957359082825
```

So, for this example, predicting `GO:0005634` is worth the most.

One more thing to mention is that GO terms form a graph and have an `is_a` relationship. For example, 

`GO:0006366`
->`is_a: GO:0006351`

->`is_a: GO:0032774`

->`is_a: GO:0016070`

->`is_a: GO:0090304`

->`is_a: GO:0006139`

->`is_a: GO:0044238`

->`is_a: GO:0008152`

->`is_a: GO:0009987`

->`is_a: GO:0008150` (the root)

If we look up `GO:0008150`, we find this:
 
```
[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
alt_id: GO:0008151
alt_id: GO:0044763
alt_id: GO:0050875
def: "Any process that is carried out at the cellular level, but not necessarily restricted to a single cell. For example, cell communication occurs among more than one cell, but occurs at the cellular level." [GOC:go_curators, GOC:isa_complete]
comment: This term should not be used for direct annotation. It should be possible to make a more specific annotation to one of the children of this term.
subset: gocheck_do_not_annotate
subset: goslim_plant
synonym: "cell growth and/or maintenance" NARROW []
synonym: "cell physiology" EXACT []
synonym: "cellular physiological process" EXACT []
synonym: "single-organism cellular process" RELATED []
is_a: GO:0008150 ! biological_process
```

Which basically means "something that does something with a cell". Needless, to say, the weight is pretty low for this one (`GO:0008510	0.0`), so don't expect a pad on the shoulder if you predict it. (It actually tells you so in the description: _It should be possible to make a more specific annotation to one of the children of this term._)

This means that the further away we are from the root (in general), the more specific our prediction is (and thus more useful). Note, that a protein can (and usually has) multiple GO terms and GO terms can (and often have) multiple parents.

## Scraping Kaggle Discussions

One thing you should definitely take advantage of is all the data in the _discussion_ tab, but usually, there is a lot. Historically, you would just browser through these, pick out the few with the most upvotes, read them and gain some insight. But it's 2025 and the year of LLMs, and we can make this process a bit more efficient.

For that purpose, I wrote a little script that would browse through all the discussions and save them locally. I could then use those, pass them into my LLM of choice and ask it nicely to summarise the whole thing for me.

This is the script I wrote:

<details>
<summary> Toggle me, if you're curious </summary>

```python 
import os
import re
from typing import cast

from fire import Fire
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.kaggle.com/competitions"
TABS: list[str] = [
    "overview",
    "code",
    "data",
    "discussion",
]


def _get_notebook_code(page) -> str:
    iframe_selector = "#rendered-kernel-content"
    try:
        page.wait_for_selector(iframe_selector, timeout=30000)

        frame = page.frame_locator(iframe_selector)

        frame.locator("body").wait_for(timeout=30000)

        code_cells = frame.locator(".jp-InputArea")

        if code_cells.count() == 0:
            code_cells = frame.locator(".code-pane, pre")

        if code_cells.count() > 0:
            return "\n\n# -------------------- CELL --------------------\n".join(
                code_cells.all_text_contents()
            )

        return "# No code content found (empty or unknown format)"

    except Exception as e:
        print(f"Error extracting code: {e}")
        return ""


def _get_all_texts(text_elements_container) -> str:
    elements = text_elements_container.locator("h1, h2, h3, h4, h5, h6, p")

    all_text = []
    for el in elements.all():
        text = el.text_content()
        if text:
            all_text.append(text.strip())

    combined_content = "\n".join(all_text)
    combined_content = cast(str, combined_content)
    return combined_content


def scrape(competition_name: str):
    os.makedirs(competition_name, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for tab in TABS:
            url = f"{BASE_URL}/{competition_name}/{tab}"
            if tab == "overview":
                page.goto(url)
                combined_content = _get_all_texts(page)
                with open(f"{competition_name}/{tab}", "wb") as f:
                    f.write(combined_content.encode("utf-8"))
            elif tab == "code":
                url += "?sortBy=voteCount&excludeNonAccessedDatasources=true"
                page.goto(url)

                code_container = page.get_by_test_id(
                    "competition-detail-code-tab-render-tid"
                )
                list_selector = code_container.locator("ul.MuiList-root.km-list")
                list_selector.wait_for()

                last_count = 0
                while True:
                    items = list_selector.locator("li")
                    current_count = items.count()

                    if current_count == last_count:
                        break

                    print(f"Loaded {current_count} notebooks so far...")
                    items.nth(current_count - 1).scroll_into_view_if_needed()
                    last_count = current_count
                    page.wait_for_timeout(3000)

                notebook_urls = []
                for i in range(current_count):
                    item = items.nth(i)
                    comment_link = item.locator("a[href*='/comments']").first

                    if comment_link.count() > 0:
                        full_url = comment_link.get_attribute("href")
                        clean_url = full_url.replace("/comments", "")
                        notebook_urls.append(f"https://www.kaggle.com{clean_url}")

                print(f"Found {len(notebook_urls)} notebooks. Scraping content...")

                for nb_url in notebook_urls:
                    print(f"Processing: {nb_url}")
                    try:
                        page.goto(nb_url)
                        page.wait_for_load_state("domcontentloaded")

                        code_content = _get_notebook_code(page)

                        if code_content:
                            slug = nb_url.split("/")[-1]
                            with open(
                                f"{competition_name}/notebooks-{slug}", "wb"
                            ) as f:
                                f.write(code_content.encode("utf-8"))
                    except Exception as e:
                        print(f"Failed {nb_url}: {e}")

            elif tab == "discussion":
                url += "?sort=votes"
                page.goto(url)

                while True:
                    next_page_button = page.get_by_role(
                        "button", name="Go to next page"
                    )
                    list_items = page.locator("ul.MuiList-root.km-list > li")
                    list_items.first.wait_for()
                    for item in list_items.all():
                        link = item.locator("a[role='link']")
                        topic_url = link.get_attribute("href")
                        if topic_url:
                            matches = re.search(
                                pattern=r"(?<=discussion\/)\d+", string=topic_url
                            )
                            if matches:
                                discussion_url = f"{BASE_URL}/{competition_name}/{tab}/{matches.group()}"
                                page.goto(discussion_url)
                                page.wait_for_load_state("networkidle")

                                text_elements_container = page.get_by_test_id(
                                    "discussion-detail-render-tid"
                                )
                                combined_content = _get_all_texts(
                                    text_elements_container
                                )

                                with open(
                                    f"{competition_name}/{tab}-{matches.group()}", "wb"
                                ) as f:
                                    f.write(combined_content.encode("utf-8"))

                                page.go_back()

                    if next_page_button.is_visible() and next_page_button.is_enabled():
                        next_page_button.click()
                        page.wait_for_load_state("networkidle")
                    else:
                        break

            elif tab == "data":
                page.goto(url)
                page.wait_for_load_state("networkidle")
                combined_content = _get_all_texts(page)
                with open(f"{competition_name}/{tab}", "wb") as f:
                    f.write(combined_content.encode("utf-8"))


if __name__ == "__main__":
    # competition_name = "cafa-6-protein-function-prediction"
    Fire(scrape)
```

</details>


Afterwards, it's just a matter of running this command

```bash 
cat cafa-6-protein-function-prediction/discussion-* | pbcopy
```

and pasting it into an LLM and ask it to summarise it.


## Standing on the Shoulders of Giants 

There is nothing new under the sun. And as such, this problem isn't new either. Smart people sat down and tackled this problem and one result of this is the [DeepGO-SE model](https://www.biorxiv.org/content/10.1101/2023.09.26.559473v1). This model - given a protein sequence - predicts their GO terms. Pretty much exactly what we are looking for. 

A quick chat with Mr. Claude helped me understand what this model does:

This model embeds the protein sequence using ESM2 (we're using `ESM-C` later) and projects those embeddings (of size 2560) and mean pool those and then project it into a different embedding space (let's call this just $d$) using a simple MLP. In the end, you just have a vector $d$. 

Then they said: each GO term is a "ball" in the $d$ dimensional vector space and it has a radius. If my protein falls e.g. into the "kinase activity ball", then it likely has that GO term. The real kicker is that they enforce the GO graph structure (because GO is a DAG, a directed, acyclic graph), so this $d$ dimensional vector space in which these balls reside MUST follow the same GO term rules. 

This means that the "kinase activity ball" must be in the "catalytic ball". From this we can infer that the protein has both of those GO terms. They enforce this behaviour through their loss function. The model basically looks like this

```python 
class GOTermModel(eqx.Module):
    centers: eqx.nn.Embedding   # (n_go_terms, d)
    radii: eqx.nn.Embedding     # (n_go_terms, 1)
    relation: Array             # (d,)
```

And the loss function is essentially this 

```
score = sigmoid(protein_vec · (relation + center) + radius)

prediction_loss = binary_cross_entropy(score, label)
total_loss = prediction_loss + λ * axiom_loss
```

The axiom loss is computed based on the output of the `GOTermModel`. They loop over each axiom (pairs of GO terms) and check if those pairs satisfy the GO term constraints and if not they get whacked with a loss which is balanced with the lambda term. This way they force the embeddings to follow the GO term structure, otherwise they will suffer a penalty.

Lastly, they train N different versions of this with different initial values. Each of those is "valid" and the more models say that a protein falls into a particular ball, the more likely it is that it is true.

This is actually pretty cool and we could train our own version of this another day, but for now, we will just use this.

The first thing we will do is let this model run over all sequences in the test superset and we will submit this directly without any modifications. This will give us both a ceiling and a floor if we use this as an "arm" in our ensemble. The ceiling is: DeepGO-SE without modifications will be at least _X_ good and the floor is: DeepGO-SE should not get worse than _X_ if we add other arms.

I ran this model over 100 sequences and it took around 35 seconds. Some napkin math tells us that - because we have roughly 240k protein sequences, that the whole generation will take around 24 hours. So I split the test set into 24 batches, one for every hour and let my PC generate all the sequences over night. I did this to make sure that it wouldn't crash in the middle and then I'd have to start again. 

Ok so, now we wait and watch our electricity bills and the room temperature rise. We will submit this once it's done.

In the meantime...

## Training a Model 

OK. We have some idea about what this challenge is about, we looked through the data and now, I believe, we are ready to create some models. Let's brainstorm a few ideas. 

For one, we have protein sequences. Proteins are 3D objects, but the sequences are 1D strings. Their properties come from their shape! Shape is EVERYTHING in cellular biology. Problem is: we can't just easily infer the shape given the string. We CAN use ESM-Fold or AlphaFold to generate them. But given that I have just 1 underpaid GPU and the challenge is only running for a couple of weeks, this option is not valid. It'd take me months to generate all the structures I'd need and even then it's not guaranteed that I could use them effectively in the model.

So instead, we use plan B: embeddings! Luckily, there are a couple of models on the market that can generate rich embedding vecotrs for a given protein sequence. Those embeddings contain lots of information about the protein and they are a helpful shortcut to getting meaningful features out of the protein sequences.

Two encoders I know of are ESM-C 300 (and 600)M and Prot-T5. Both models are available online and you can easily download and use them.

So the first thing I did was to compute all the embeddings for all the proteins in the training data. I stored them in `float16` because I need my money for food and have non left for storage.

I also saved them in 2 ways: one where I store them "raw" and another is where I store their "mean" across the sequence length dimension. Here's an example:

Suppose you have a protein that is 50 amino acids long. You use ESM-C 600M. The model will output a matrix of shape $50 \times 1154$, where 50 is the sequence length and 1154 is the embedding dimension of ESM-C. Another protein might have a difference sequence length, thus a differently shaped matrix. In JAX land, remember, we MUST NOT HAVE dynamically shaped input matrices. So we have three options:

1) disregard everything and pass dynamic shapes in anyway and cry as your model recompiles for every shape (bad idea)
2) set a max sequence length and truncate/pad all the proteins (good idea)
3) compute the mean across `axis=0` (also good idea)

Having stored the raw embeddings (with the variable sequence lengths) alongside the mean ones, gives us the option to pursue also option 2). I repeated the process for the `Rostlab/prot_t5_xl_half_uniref50-enc` model (which I will henceforth abbreviate as just `prot_t5`). This process used up some 200GB of disk space. Oof.
