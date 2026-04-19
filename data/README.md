# Data Layout

This repository does not version the actual dataset files.

Kept in git:

- folder layout
- download and preparation instructions
- links back to the upstream Multi30k source

Excluded from git:

- raw `.gz` dataset files
- processed training text files
- generated vocab files

To recreate the local data:

```bash
python scripts/download_multi30k.py
python scripts/prepare_data.py
```

Relevant subdirectories:

- `data/raw/`: raw downloaded dataset files
- `data/processed/`: tokenized and vocabulary-built training data
