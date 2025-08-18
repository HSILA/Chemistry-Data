## ChemRxiv module

A small utility to gather ChemRxiv preprint metadata and download PDFs. It mirrors the style of the other modules in this repository and is driven by a YAML config.

### What it does
- **gather**: Calls the ChemRxiv public API, iterates all preprints, appends one JSON object per line to a JSONL file, and writes a CSV for convenient downstream use.
- **download**: Reads the JSONL and downloads missing PDFs into a local folder, showing an accurate tqdm progress bar and respecting cooldowns.

### Directory layout
- `ChemRxiv/chemrxiv.py`: CLI entrypoint (gather/download stages)
- `ChemRxiv/config.py`: Pydantic configuration models
- `ChemRxiv/configs/chemrxiv_default.yaml`: Example configuration

### Installation
Use the repository’s virtual environment or install the minimal dependencies for this module:
```bash
pip install -r requirements.txt
```

### Configuration
Edit `ChemRxiv/configs/chemrxiv_default.yaml` (copy it if you need variants).

- gather:
  - `jsonl_path`: Path to append JSONL metadata (one object per line)
  - `csv_path`: Path to write a CSV summary of all preprints
  - `batch_size`: Number of new rows to process before a longer pause
  - `batch_delay`: Pause (seconds) after each batch completes
  - `cooldown_time`: Long cooldown (seconds) applied after connection errors before resuming
  - `request_delay`: Short delay (seconds) between consecutive API requests to avoid rate limits

- download:
  - `jsonl_path`: Path to the JSONL produced in the gather stage
  - `download_dir`: Folder to store PDFs
  - `batch_size`: Number of successful downloads before a longer pause
  - `batch_delay`: Pause (seconds) after each batch completes
  - `cooldown_time`: Long cooldown (seconds) applied after connection errors before resuming
  - `request_delay`: Short delay (seconds) between consecutive downloads

### Running
From the repo root:
```bash
python ChemRxiv/chemrxiv.py --config ChemRxiv/configs/chemrxiv_default.yaml --stage gather
python ChemRxiv/chemrxiv.py --config ChemRxiv/configs/chemrxiv_default.yaml --stage download
```

### CSV schema
The gather stage writes a CSV with the following columns:

- `id`: ChemRxiv internal identifier for the item.
- `doi`: DOI string of the preprint (e.g., `10.26434/chemrxiv-2025-xxxx`).
- `title`: The preprint title text.
- `abstract`: Abstract text as returned by the API.
- `publishedDate`: ISO timestamp when the preprint version was published.
- `submittedDate`: ISO timestamp when the preprint was originally submitted.
- `status`: Item status string (e.g., `PUBLISHED`).
- `version`: Version number string of this item (e.g., `1`, `2`).
- `license`: Human-readable license name (e.g., `CC BY 4.0`).
- `keywords`: Semicolon-separated list of keyword strings.
- `authors`: Semicolon-separated list of author names in `First Last` format.
- `pdf_url`: Direct URL to the main PDF (when present in the response asset block).

Notes:
- `keywords` and `authors` are flattened into semicolon-separated strings for convenience.
- Some items may not have a `pdf_url` (rare); those are skipped by the downloader.

### JSONL contents
- Each line is a JSON object mirroring the ChemRxiv `itemHits` record (with its nested `item` structure). Keeping the raw JSON allows flexible re-parsing in the future without re-calling the API.

### Deduplication and resume behavior
- Download stage computes a fixed list of download candidates by comparing JSONL entries to existing files in `download_dir`.

### Rate limiting and robustness
- Use `request_delay` to throttle per-request cadence.
- Use `batch_size` + `batch_delay` to add longer periodic pauses.
- On connection errors, the script applies `cooldown_time` before continuing.

### Troubleshooting
- If you see transient JSON decode or content decoding errors, increase `request_delay` and/or `batch_delay`.
- Ensure you are executing from the repository root so relative paths in the YAML resolve correctly.

### GROBID (PDF → TEI XML)

After downloading PDFs, to extract textual paragraphs we use GROBID (GeneRation Of BIbliographic Data), a machine‑learning tool for parsing scholarly PDFs into structured TEI XML suitable for text mining and downstream analysis. It converts PDF layout into machine‑readable XML that we will use to extract paragraphs. See the official introduction for background: [GROBID documentation](https://grobid.readthedocs.io/en/update-documentation/Introduction/).

GROBID runs as a server; the Python client and CLI are lightweight HTTP wrappers. Start the server, then invoke the client from your virtual environment.

1) Start the GROBID server (Docker)

```bash
docker run --rm --init -p 8070:8070 grobid/grobid:0.8.2
```
docker run -d --name grobid --init -p 8070:8070 grobid/grobid:0.8.1

Replace `0.8.2` with the current tag if newer. This exposes the API on `http://localhost:8070`.

Sanity check (in another shell):

```bash
curl http://localhost:8070/api/isalive
curl http://localhost:8070/api/version
```

If you run a client from another container, don’t use `localhost`; use the container/service name, e.g. `http://grobid:8070` on the same Docker network.

2) Convert a directory of PDFs to TEI XML (CLI recommended)

From your virtualenv where `grobid-client-python` is installed:

```bash
grobid_client \
  --config ChemRxiv/grobid_config.json \
  --input /path/to/pdfs_in \
  --output /path/to/tei_out \
  --n 8 \
  --teiCoordinates \
  --segmentSentences \
  processFulltextDocument
```

Notes:
- `processFulltextDocument` turns full PDFs into TEI XML. Other services include `processHeaderDocument` and `processReferences`.
- `--input` must be a directory of PDFs (recursively walked). `--output` receives `*.grobid.tei.xml`. If `--output` is omitted, XML is written next to each PDF.
- `ChemRxiv/grobid_config.json` sets the server URL and retry/backoff behavior. The `coordinates` list only applies if `--teiCoordinates` is passed (as shown) and includes paragraph and sentence elements useful for paragraph extraction.
- `--n` controls parallel requests. If you see HTTP 503s, lower `--n` or increase the sleep/backoff in the config; the client will wait and retry automatically.

3) Minimal curl test (single file)

```bash
curl -s -H "Accept: application/xml" \
  -F "input=@./one.pdf" \
  http://localhost:8070/api/processFulltextDocument > one.grobid.tei.xml
```

4) Common gotchas
- **Server not up**: `/api/isalive` returns false or times out → (re)start Docker and check port 8070 mapping.
- **HTTP 503**: Too many parallel requests → reduce `--n`.
- **HTTP 500**: Inspect server logs (e.g., `logs/grobid-service.log` in the container).
- **Containers**: Use the service name instead of `localhost` when calling across containers.

### Parse TEI XML to paragraphs (optional: push to Hugging Face)

With TEI XML produced by GROBID, use `ChemRxiv/parse_grobid.py` to extract textual paragraphs and optionally publish a dataset to the Hugging Face Hub.

What it does
- Walks a directory of `*.xml` files (including `*.tei.xml`).
- Extracts paragraph text from TEI body divisions while skipping inline `<ref>` content.
- Normalizes text and filters out very short passages and low‑likelihood text based on a unigram model.
- Writes per‑document JSON files containing the resulting paragraphs and can push a consolidated dataset to the Hub.

Basic usage (from repo root)
```bash
python ChemRxiv/parse_grobid.py \
  --src /path/to/tei_out \
  --dst /path/to/paragraphs_json \
  --concat-p
```

Push to Hugging Face (optional)
```bash
python ChemRxiv/parse_grobid.py \
  --src /path/to/tei_out \
  --dst /path/to/paragraphs_json \
  --hf-path your-username/chemrxiv-paragraphs \
  --hf-config default \
  --concat-p
```

Notes
- `--src` is the directory of TEI XML files output by GROBID.
- `--dst` receives per‑document JSON files named `<docid>-paragraphs.json`.
- `--concat-p` concatenates all `<p>` elements per TEI `<div>` into one paragraph; omit it to keep individual `<p>` elements as separate entries.
- For pushing to the Hub, ensure you are authenticated with `huggingface-cli login` and have permission to write to `--hf-path`.

### License and usage
- Respect the ChemRxiv API usage and rate-limit guidelines.
- The generated data and PDFs are subject to the licenses indicated by each item’s metadata and content license.

