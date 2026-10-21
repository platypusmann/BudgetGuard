# Data and licenses

The code in this repo is MIT licensed, see `LICENSE`. The data is a different story, because most of it comes from somewhere else.

## AmbigNQ, CC BY-SA 3.0

The questions, gold answers and ambiguity annotations come from AmbigNQ, the light config of the AmbigQA dataset:

- Sewon Min, Julian Michael, Hannaneh Hajishirzi and Luke Zettlemoyer. "AmbigQA: Answering Ambiguous Open-domain Questions." EMNLP 2020.
- Hugging Face: `sewon/ambig_qa`. Project page: https://github.com/shmsw25/AmbigQA
- License: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0), https://creativecommons.org/licenses/by-sa/3.0/
- AmbigNQ is built on the NQ-Open questions from Natural Questions (Kwiatkowski et al., 2019), with answers drawn from Wikipedia.

Check the dataset card for the current terms, since that is the authority and not this note.

## What in this repo is derived from it

- `data/splits/` holds the questions, gold answers and interpretations from AmbigNQ, plus my changes: a hash-based calib and tune split of the official train set, the official validation set as test, a 300-question pilot, one interpretation chosen by hash as the user's intent, and the prompt list.
- `data/cache/` holds model answers to those questions, and `data/graded/`, `results/` and the other analysis files are computed from them.
- The synthetic data in `src/budget_guard/synthetic.py` generates invented questions and invented answers. It copies the format of AmbigNQ and none of its content.

I treat the questions, gold answers and everything computed directly from them as adaptations of AmbigNQ. That material is shared under CC BY-SA 3.0, with attribution to the authors above, and any derivative of it has to be shared under the same license. It is not covered by the MIT license. I changed the data as described in the list above.

## Models

- Tier 0 is `Qwen/Qwen3-4B-Instruct-2507`, run on Kaggle. See its model card for its license.
- Tier 1 and Tier 2 are `gpt-6-luna` and `gpt-6-sol`, called through the OpenAI API. Their outputs in `data/cache/` are subject to OpenAI's terms of use.

None of this is legal advice. If you reuse the data, read the licenses above.
