# Third-party notices

The MIT License in the repository root applies to original scKITE source code.
Third-party components remain subject to their respective licenses.

## GEARS

The directory `downstream/gene_perturbation_prediction/gears/` contains code
derived from or compatible with GEARS:

- Project: GEARS
- Source: <https://github.com/snap-stanford/GEARS>
- Local package version marker: 0.1.2
- License: MIT
- Copyright: Copyright (c) 2022 Yusuf Roohani, Kexin Huang, Jure Leskovec

The upstream license text is retained in
[`downstream/gene_perturbation_prediction/gears/LICENSE`](downstream/gene_perturbation_prediction/gears/LICENSE).

The GEARS license permits use, modification, and redistribution provided that
its copyright and permission notices are retained. The local copy includes
modifications for scKITE and other single-cell foundation-model adapters. The
exact upstream commit was not recorded in the original import history and must
be identified before a provenance-complete release.

## External models and libraries

The repository provides integration code for scGPT and scFoundation but does
not redistribute their model weights. Users must obtain external software and
weights from their official sources and comply with the applicable licenses
and terms.

## Data and biological resources

The source-code license does not grant additional rights to datasets, model
weights, pretrained tokenizers, or third-party biological resources. Their
provenance, redistribution terms, and licenses must be checked separately.
