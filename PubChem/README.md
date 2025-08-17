# PubChem Data Retrieval and Processing

This module retrieves compound records from PubChem and processes the downloaded JSON files into two CSVs: compounds metadata and descriptions with references.

## Structure

- `config.py`: Pydantic config models.
- `pubchem.py`: Single entry point to run download or parse stages using a YAML config.
- `configs/`: Example YAML configuration files.
- `data/`: Suggested default directory for downloaded JSONs (created at runtime).

## Install

```
pip install -r requirements.txt
```

## Configure

Copy and adjust `PubChem/configs/pubchem_default.yaml`.

- `download.predefined_cids_path`: If set, points to a JSON file containing a list of integer CIDs to download. When provided, the downloader fetches only these CIDs.
- Otherwise, it will determine the last downloaded CID in `download.save_path` and continue sequentially up to `download.max_cid`.

## Usage

Download sequentially (continue from last):

```
python PubChem/pubchem.py --config PubChem/configs/pubchem_default.yaml --stage download
```


Process JSONs to CSVs:

```
python PubChem/pubchem.py --config PubChem/configs/pubchem_default.yaml --stage parse
```


### Retrieved fields per compound

From the PubChem record (primarily the `Names and Identifiers` section):

- **CID**: Numeric compound identifier (`Record.RecordNumber`).
- **Title**: Compound title (`Record.RecordTitle`).
- **MolecularFormula**: Molecular formula string.
- **IUPACName**: From `Computed Descriptors`.
- **InChI**: From `Computed Descriptors`.
- **SMILES**: From `Computed Descriptors`.
- **Synonyms**: From `Synonyms` → `Depositor-Supplied Synonyms`, filtered to textual entries (must contain letters, exclude strings with two consecutive digits), and length between 3 and 105 characters.

Additionally, description entries are extracted from `Record Description`, which consists of compound descriptions from various sources.

### CSV outputs

- **compounds.csv** columns:
  - `CID`
  - `Title`
  - `MolecularFormula`
  - `IUPACName`
  - `InChI`
  - `SMILES`
  - `Synonyms` (list serialized as a string in the CSV)

- **descriptions.csv** columns:
  - `CID`
  - `Title`
  - `Description` (text from `Record Description`)
  - `ReferenceNumber`
  - `SourceName`
  - `SourceID`
  - `ReferenceDescription`
  - `URL`

