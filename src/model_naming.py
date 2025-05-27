
MODEL_ID_TO_SHORT_NAME = {
    'chs20/FuseLIP-S-CC3M-MM':
        'chs20/FuseLIP-S-CC3M-MM',
    'chs20/FuseLIP-B-CC3M-MM':
        'chs20/FuseLIP-B-CC3M-MM',
    'chs20/FuseLIP-S-CC12M-MM':
        'chs20/FuseLIP-S-CC12M-MM',
    'chs20/FuseLIP-B-CC12M-MM':
        'chs20/FuseLIP-B-CC12M-MM',
}

_MODEL_SHORT_NAME_TO_ID = {v: k for k, v in MODEL_ID_TO_SHORT_NAME.items()}
assert len(_MODEL_SHORT_NAME_TO_ID) == len(MODEL_ID_TO_SHORT_NAME)

