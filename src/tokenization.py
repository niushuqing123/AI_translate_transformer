from __future__ import annotations

import re
from typing import Iterable, List

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|[^\w\s]", flags=re.UNICODE)


def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text.strip())


def detokenize(tokens: Iterable[str]) -> str:
    text = ' '.join(tokens)
    for mark in [',', '.', '!', '?', ';', ':', '%', ')', ']', '}']:
        text = text.replace(' ' + mark, mark)
    for mark in ['(', '[', '{', '$', '€']:
        text = text.replace(mark + ' ', mark)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    text = text.replace(" - ", "-")
    return ' '.join(text.split())
