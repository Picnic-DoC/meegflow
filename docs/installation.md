# Installation

## From PyPI

```bash
pip install meegflow
```

## From source

```bash
git clone https://github.com/Picnic-DoC/meegflow.git
cd meegflow
pip install -e .
```

## Docker

The Docker image bundles all system-level dependencies (useful on HPC clusters or when MNE's compiled extensions are hard to install).

```bash
git clone https://github.com/Picnic-DoC/meegflow.git
cd meegflow
docker build -t meegflow .

docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/config.yaml:/config.yaml \
    meegflow \
    --bids-root /data \
    --config /config.yaml \
    --subjects 01 02 \
    --tasks rest
```

## Requirements

- Python ≥ 3.8
- MNE-Python
- MNE-BIDS
- PyYAML
- Rich
