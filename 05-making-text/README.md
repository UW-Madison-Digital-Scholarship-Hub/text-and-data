# Making Text—Computational Language Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UW-Madison-Digital-Scholarship-Hub/text-and-data/blob/main/05-making-text/Making-Text.ipynb)

## Description
How does a machine generate text, rather than just analyze it? This session explores how language models work, from simple n-gram prediction to the transformer systems behind today's LLMs. You'll get hands-on time with a small, locally-hosted model and see what changes when one is trained on a specific collection. The goal is demystification: connecting text generation back to how text gets digitized, broken apart, and represented in the first place.

## Details
When: Tuesday, November 17, 2026 | 1-2:30 p.m
Where: Memorial Library 231 (second floor computer classroom)

## Notes
The notebook builds an n-gram model from the eight books saved in `../03-Tokenization/texts/`, then runs GPT-2 through Hugging Face `transformers`. Colab's default CPU runtime is sufficient; the GPT-2 download is roughly 500 MB on first run.
