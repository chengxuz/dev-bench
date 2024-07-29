# DevBench: A multimodal developmental benchmark for language learning

This is the project repository for DevBench ([preprint](https://doi.org/10.48550/arXiv.2406.10215)), a multimodal benchmark intended to assess Vision--Language Models in terms of their similarities with human responses across development.

Evaluation of an implemented model against DevBench can be conducted as follows:
```
> python eval.py model_name
```
where `model_name` is one of `clip_base`, `clip_large`, `blip`, `flava`, `bridgetower`, `vilt`, or `cvcl`.

You can add additional models by constructing a subclass with methods for obtaining image features, text features, and image--text similarity scores.

NOTE: Assets and human data are not currently available as some of them come from unpublished datasets; we anticipate adding download scripts as soon as the embargo on these are lifted.
