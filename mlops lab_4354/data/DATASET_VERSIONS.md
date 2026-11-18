# Dataset Version Manifest

| Version | File | Records | Simulation |
|---|---|---:|---|
| v1 | `data/raw/manuscript_rooms_v1.csv` | 250 | Baseline storage conditions |
| v2 | `data/raw/manuscript_rooms_v2.csv` | 250 | Monsoon conditions: higher humidity, more door openings, lower ventilation efficiency |

The combined file is `data/raw/manuscript_rooms.csv`. Use
`python src/data_generation.py --n 250` to regenerate deterministic versions.
