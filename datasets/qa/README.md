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
