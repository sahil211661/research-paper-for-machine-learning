# Literature review (verified sources only)

The full comparison table (title, authors, year, venue, DOI/URL, main method, evaluation setting, relevant findings, limitations)
is in notebook **Section 3**; bibliographic entries are in `references.bib`. Verification method: each entry was checked during
this project against at least one publisher, repository or index record found by web search (title, authors, year, venue, DOI).
Details of a paper that were not visible in the checked records are marked *not verified* (e.g. the abstract of Barros & Santos 2018).

## What the checked literature covers
* Surveys/reviews [Gama 2014; Lu 2019]: concept drift = change of the relation between inputs and target; detection, understanding and adaptation are separate tasks.
* Detectors [Bifet & Gavaldà 2007 (ADWIN); Page 1954 and Hinkley 1971 (cumulative-sum / Page-Hinkley)].
* Evaluation [Gama, Sebastião & Rodrigues 2013]: prequential error with forgetting (windows / fading factors).
* Detector benchmarking [Gonçalves et al. 2014; Barros & Santos 2018].
* Generator, learner, software [Street & Kim 2001 (SEA); Domingos & Hulten 2000 (Hoeffding trees); Montiel et al. 2021 (River)].

## Research gap (conservative)
This was not an exhaustive systematic review; **no novelty claim is made**. The project contributes a reproducible, laptop-scale factorial comparison
(detector x sensitivity x trigger x action, identical streams, paired seeds) with explicit separation of concept change, prevalence change and covariate shift,
a pre-specified alarm-matching rule and recovery definition, cost measurements and Holm-corrected paired statistics.

## NOT verified (excluded from the bibliography; verify before citing)
Demšar (2006); Webb et al. (2016); Bayram, Ahmed & Kassler (2022); Hoens, Polikar & Chawla (2012); Gama et al. (2004, DDM);
Baena-García et al. (2006, EDDM); Harries (1999, SPLICE-2 report).
