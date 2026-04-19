# Raw Data

The raw dataset files are intentionally excluded from version control.

Expected local source:

- `data/raw/multi30k/train.en.gz`
- `data/raw/multi30k/train.de.gz`
- `data/raw/multi30k/val.en.gz`
- `data/raw/multi30k/val.de.gz`
- `data/raw/multi30k/test_2016_flickr.en.gz`
- `data/raw/multi30k/test_2016_flickr.de.gz`

Download method:

```bash
python scripts/download_multi30k.py
```

Upstream references:

- Repository: `https://github.com/multi30k/dataset`
- Task 1 raw files: `https://github.com/multi30k/dataset/tree/master/data/task1/raw`
- Raw content base used by this project: `https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/`
