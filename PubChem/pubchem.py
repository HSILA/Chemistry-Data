
import requests
import json
import time
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import tqdm
import pandas as pd
import argparse
import re
import yaml
from pydantic import BaseModel, Field
from typing import Optional


class DownloadConfig(BaseModel):
    max_cid: int = Field(..., description="Maximum compound CID to download")
    save_path: str = Field(..., description="Directory to save downloaded JSON files")
    batch_size: int = Field(1000, description="Number of requests per batch")
    batch_delay: int = Field(120, description="Delay in seconds after retrieving each batch")
    cooldown_time: int = Field(120, description="Cooldown time in seconds on connection errors")
    request_delay: int = Field(3, description="Delay in seconds between consecutive requests")
    predefined_cids_path: Optional[str] = Field(
        None,
        description="Path to a JSON file containing a list of CIDs to retrieve. If provided, downloader only retrieves these CIDs.",
    )


class ParseConfig(BaseModel):
    jsons_dir: str = Field("./PubChem", description="Directory containing downloaded JSON files")
    comp_csv: str = Field("compounds.csv", description="CSV file to output compounds data")
    desc_csv: str = Field("descriptions.csv", description="CSV file to output descriptions data")
    batch_size: int = Field(1000, description="Batch size for processing JSON files")


class Config(BaseModel):
    download: DownloadConfig = Field(..., description="Download configuration")
    parse: ParseConfig = Field(..., description="Parsing configuration")


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=64),
    stop=stop_after_attempt(5),
)
def robust_get(url):
    return requests.get(url, timeout=10)


def get_last_cid(save_path):
    files = [f for f in os.listdir(save_path) if f.startswith("cid_") and f.endswith(".json")]
    if not files:
        return 0
    cids = [int(f.split("_")[1].split(".")[0]) for f in files]
    return max(cids)


def download_pubchem(max_cid, save_path, batch_size, batch_delay, cooldown_time, request_delay, predefined_cids_path: Optional[str] = None):
    os.makedirs(save_path, exist_ok=True)
    num_requests = 0

    predefined_cids = None
    total = None
    if predefined_cids_path:
        try:
            with open(predefined_cids_path, "r") as f:
                predefined_cids = json.load(f)
            if not isinstance(predefined_cids, list):
                raise ValueError("predefined_cids JSON must be a list of integers")
            print(f"Using predefined CIDs list with {len(predefined_cids)} entries.")
            total = len(predefined_cids)
        except Exception as e:
            raise RuntimeError(f"Failed to read predefined CIDs from {predefined_cids_path}: {e}")
    else:
        last_cid = get_last_cid(save_path)
        message = (
            "Previous download not found. Starting from scratch."
            if last_cid == 0
            else f"Start downloading from CID: {last_cid}"
        )
        print(message)
        start_cid = last_cid + 1
        total = (max_cid - start_cid + 1)

    cid_iterable = predefined_cids if predefined_cids is not None else range(start_cid, max_cid + 1)

    for cid in tqdm.tqdm(cid_iterable, total=total):
        while True:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
            try:
                response = robust_get(url)
                if response.status_code == 200:
                    data = response.json()
                    with open(os.path.join(save_path, f"cid_{cid}.json"), "w") as f:
                        json.dump(data, f)
                    num_requests += 1
                break
            except requests.ConnectionError as e:
                print(
                    f"Failed to fetch CID {cid} after retries: {e}. Waiting {cooldown_time}s before continuing.")
                time.sleep(cooldown_time)
            except Exception as e:
                print(f"Unexpected error for CID {cid}: {e}. Skipping.")
                break
        time.sleep(request_delay)
        if num_requests > 0 and num_requests % batch_size == 0:
            print(f"\nBatch complete: {num_requests} requests sent. Waiting {batch_delay}s...")
            time.sleep(batch_delay)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_path_jsons(directory):
    for file in sorted(os.listdir(directory), key=natural_sort_key):
        if file.endswith(".json"):
            yield read_json(os.path.join(directory, file))


def is_textual(s):
    if re.search(r"\d\d", s):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    return True


def append_to_csv(df, csv_path):
    with open(csv_path, "a") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def get_references(data):
    refs = {}
    for r in data["Record"]["Reference"]:
        refs[r["ReferenceNumber"]] = r
    return refs


def get_section(sections, key):
    for sec in sections:
        if sec["TOCHeading"] == key:
            return sec
    return None


def get_descriptions(names_identifiers_section):
    ref_desc = {}
    records_desc = get_section(names_identifiers_section["Section"], "Record Description")
    if records_desc is None:
        return ref_desc
    for item in records_desc["Information"]:
        ref_number = item["ReferenceNumber"]
        if ref_number != 111:
            ref_desc[ref_number] = item["Value"]["StringWithMarkup"][0]["String"]
    return ref_desc


def get_descriptors(names_identifiers_section):
    computed_descriptors = get_section(names_identifiers_section["Section"], "Computed Descriptors")
    iupac_name = smiles = inchi = None
    for item in computed_descriptors["Section"]:
        if item["TOCHeading"] == "IUPAC Name":
            iupac_name = item["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
        elif item["TOCHeading"] == "InChI":
            inchi = item["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
        elif item["TOCHeading"] == "SMILES":
            smiles = item["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
    return iupac_name, smiles, inchi


def get_molecular_formula(names_identifiers_section):
    molecular_formula = get_section(names_identifiers_section["Section"], "Molecular Formula")
    return molecular_formula["Information"][0]["Value"]["StringWithMarkup"][0]["String"]


def get_synonyms(names_identifiers_section):
    try:
        synonyms_sec = get_section(names_identifiers_section["Section"], "Synonyms")
        synonyms = get_section(synonyms_sec["Section"], "Depositor-Supplied Synonyms")
        syns_list = synonyms["Information"][0]["Value"]["StringWithMarkup"]
        return [syn["String"] for syn in syns_list]
    except:
        return []


def process_json_files(jsons_dir, comp_csv, desc_csv, batch_size):
    comp_rows, desc_rows = [], []
    json_files = read_path_jsons(jsons_dir)
    num_jsons = len([f for f in os.listdir(jsons_dir) if f.endswith(".json")])

    for i, data in enumerate(tqdm.tqdm(json_files, total=num_jsons)):
        if i % batch_size == 0 and i != 0:
            append_to_csv(pd.DataFrame(comp_rows), comp_csv)
            append_to_csv(pd.DataFrame(desc_rows), desc_csv)
            comp_rows, desc_rows = [], []

        cid = data["Record"]["RecordNumber"]
        title = data["Record"]["RecordTitle"]
        references = get_references(data)
        main_sections = data["Record"]["Section"]
        name_ids_section = get_section(main_sections, "Names and Identifiers")
        descriptions = get_descriptions(name_ids_section)
        iupac_name, smiles, inchi = get_descriptors(name_ids_section)
        molecular_formula = get_molecular_formula(name_ids_section)
        synonyms = list(filter(is_textual, get_synonyms(name_ids_section)))
        synonyms = [syn for syn in synonyms if 3 <= len(syn) <= 105]

        comp_row = {
            "CID": cid,
            "Title": title,
            "MolecularFormula": molecular_formula,
            "IUPACName": iupac_name,
            "InChI": inchi,
            "SMILES": smiles,
            "Synonyms": synonyms,
        }
        comp_rows.append(comp_row)

        if descriptions:
            for ref_id, desc in descriptions.items():
                ref = references[ref_id]
                desc_row = {
                    "CID": cid,
                    "Title": title,
                    "Description": desc,
                    "ReferenceNumber": ref_id,
                    "SourceName": ref["SourceName"],
                    "SourceID": ref["SourceID"],
                    "ReferenceDescription": ref["Description"],
                    "URL": ref["URL"],
                }
                desc_rows.append(desc_row)

    # Flush remaining rows
    if comp_rows:
        append_to_csv(pd.DataFrame(comp_rows), comp_csv)
    if desc_rows:
        append_to_csv(pd.DataFrame(desc_rows), desc_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PubChem downloader and parser using YAML config with Pydantic")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stage", required=True,
                        choices=["download", "parse"], help="Stage to run: download or parse")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        raw_config = yaml.safe_load(file)

    # Validate and populate our Pydantic config
    config_obj = Config(**raw_config)

    if args.stage == "download":
        dl = config_obj.download
        download_pubchem(
            max_cid=dl.max_cid,
            save_path=dl.save_path,
            batch_size=dl.batch_size,
            batch_delay=dl.batch_delay,
            cooldown_time=dl.cooldown_time,
            request_delay=dl.request_delay,
            predefined_cids_path=dl.predefined_cids_path,
        )
    elif args.stage == "parse":
        pr = config_obj.parse
        process_json_files(
            jsons_dir=pr.jsons_dir,
            comp_csv=pr.comp_csv,
            desc_csv=pr.desc_csv,
            batch_size=pr.batch_size
        )
