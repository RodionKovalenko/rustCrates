# Additional question-answering data

Every `.json` file in this folder (SQuAD format) is trained on together with `datasets/xquad.de.json`.
Large datasets are stored in parts of at most 10 MB so they can be pushed to and pulled from GitHub.

## Adding a dataset

1. Download the dataset's SQuAD-format JSON file (see below).
2. Split it into parts in this folder:

       .\target\release\wav-transformer.exe split-qa <downloaded.json> <name> [part_mb]

   e.g. `split-qa GermanQuAD_train.json germanquad_train` writes `germanquad_train_001.json`, `_002.json`, ...
3. Commit the parts. Questions without an answer (SQuAD 2.0 "impossible" questions) are skipped when loading.

Adding data changes which questions are held out for validation (the split is drawn over all records), so train a
new model afterwards.

## German datasets

Check the license on the source page before adding a dataset; keep its attribution here.

| Dataset | Questions (de) | Source | License |
|---|---|---|---|
| GermanQuAD (deepset) | ~13,700 (train + test) | https://huggingface.co/datasets/deepset/germanquad | CC BY 4.0 |
| MLQA, German part (Facebook Research) | ~5,000 (dev `dev-context-de-question-de.json` + test `test-context-de-question-de.json` in `MLQA_V1.zip`) | https://github.com/facebookresearch/MLQA | CC BY-SA 3.0 |
| XQuAD (de), already in `datasets/` | 1,190 | https://github.com/google-deepmind/xquad | CC BY-SA 4.0 |

## Contents of this folder

`germanquad_train_001..003.json` (11,518 questions) and `germanquad_test_001.json` (1,839 questions):
**GermanQuAD** by deepset (Möller, Risch, Pietsch 2021, arXiv:2104.12741), https://huggingface.co/datasets/deepset/germanquad,
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Changes made to the original: converted from the Hugging Face parquet export (`refs/convert/parquet`, the original
`GermanQuAD.zip` download no longer exists) to SQuAD JSON with `convert_germanquad.py`; the title line of each context
became the article title and `=== Section ===` headings plain text; only the first non-empty answer of each question
is kept; 365 test questions whose context is longer than 4,000 characters were left out; split into 2 MB parts with
`split-qa ... 2`.
