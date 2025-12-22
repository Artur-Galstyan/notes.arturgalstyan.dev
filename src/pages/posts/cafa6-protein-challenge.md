---
layout: ../../layouts/PostLayout.astro
title: Kaggle Competition CAFA 6 Protein Function Prediction (DRAFT)
date: 2025-12-22
---

# Kaggle Competition: CAFA 6 Protein Function Prediction (DRAFT)

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
