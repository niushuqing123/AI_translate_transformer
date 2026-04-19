# Processed Data

Processed dataset artifacts are generated locally and excluded from version control.

Expected local outputs include:

- tokenized `train/valid/test` source and target text
- `src_vocab.json`
- `tgt_vocab.json`
- `meta.json`

Generate them with:

```bash
python scripts/prepare_data.py
```

Prepared variants used in this project:

- `multi30k_en_de`
- `multi30k_en_de_quick`
