# Chemistry-Data

Lightweight utilities and configs to collect, parse, and prepare open chemistry text/data from ChemRxiv and PubChem, plus scripts to generate synthetic queries for retrieval tasks. The resulting datasets are used in the ChEmbed manuscript and released openly.

## Modules
- **ChemRxiv**: Download and parse ChemRxiv preprints (PDF → structured XML via GROBID), and produce paragraph corpora. See `ChemRxiv/README.md`.
- **PubChem**: Extract chemical information from PubChem PUGView endpoints and prepare clean, structured JSON/TSV outputs. See `PubChem/README.md`.
- **query-generation**: Generate high-quality synthetic queries for retrieval training and evaluation. See `query-generation/README.md`.

## Data availability
Curated and generated datasets produced by this codebase are hosted in the following collection on HuggingFace:

- Hugging Face Collection: `BASF-AI / Chemical Data` - https://huggingface.co/collections/BASF-AI/chemical-data-685b21fedf9026ead61b9f24

## Manuscript
These data are used in the following manuscript:

- Kasmaee, A. S., Khodadad, M., Astaraki, M., Saloot, M. A., Sherck, N., Mahyar, H., & Samiee, S. (2025). ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings. arXiv:2508.01643. https://doi.org/10.48550/arXiv.2508.01643

## Getting started
Install dependencies and follow the module-level READMEs for setup and usage details.

```bash
pip install -r requirements.txt
```
