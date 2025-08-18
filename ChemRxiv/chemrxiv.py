import argparse
import json
import os
import time
import csv
from typing import Dict, Any, Iterable

import requests
import tqdm
import yaml

from config import ChemRxivConfig


class ChemRxivAPI:
    base = 'https://chemrxiv.org/engage/chemrxiv/public-api/v1'
    pagesize = 50

    def request(self, url, method, params):
        if method.casefold() == 'get':
            return requests.get(url, params=params, timeout=30)
        elif method.casefold() == 'post':
            return requests.post(url, json=params, timeout=30)
        else:
            raise Exception(f'Unknown method for query: {method}')

    def query(self, query, method='get', params=None):
        r = self.request(f'{self.base}/{query}', method, params)
        r.raise_for_status()
        return r.json()

    def query_generator(self, query, method='get', params=None) -> Iterable[Dict[str, Any]]:
        if params is None:
            params = {}
        n = 0
        while True:
            page_params = {**params, 'limit': self.pagesize, 'skip': n * self.pagesize}
            r = self.request(f'{self.base}/{query}', method, page_params)
            r.raise_for_status()
            payload = r.json()
            items = payload.get('itemHits', [])
            if not items:
                return
            for item in items:
                yield item
            n += 1

    def all_preprints(self) -> Iterable[Dict[str, Any]]:
        return self.query_generator('items')

    def number_of_preprints(self) -> int:
        return self.query('items').get('totalCount', 0)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(filename: str, max_length: int = 255) -> str:
    if len(filename) > max_length:
        base, ext = os.path.splitext(filename)
        allowed = max_length - len(ext)
        base = base[:allowed]
        filename = base + ext
    return filename


def gather_metadata(cfg: ChemRxivConfig):
    g = cfg.gather
    ensure_dir(os.path.dirname(g.jsonl_path) or '.')
    ensure_dir(os.path.dirname(g.csv_path) or '.')

    api = ChemRxivAPI()
    total = api.number_of_preprints()
    generator = api.all_preprints()

    # Build a set of existing IDs to avoid duplicate JSONL/CSV rows
    existing_ids = set()
    if os.path.exists(g.jsonl_path):
        try:
            with open(g.jsonl_path, 'r') as existing:
                for line in existing:
                    try:
                        obj = json.loads(line)
                        item = obj.get('item', {})
                        pid = item.get('id')
                        if pid:
                            existing_ids.add(pid)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            existing_ids = set()

    num_written = 0
    jsonl_f = open(g.jsonl_path, 'a')

    # Columns: id, doi, title, abstract, publishedDate, submittedDate, status, version, license, keywords, authors, pdf_url
    csv_columns = [
        'id', 'doi', 'title', 'abstract', 'publishedDate', 'submittedDate', 'status',
        'version', 'license', 'keywords', 'authors', 'pdf_url'
    ]
    csv_needs_header = not os.path.exists(g.csv_path) or os.path.getsize(g.csv_path) == 0
    csv_file = open(g.csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    if csv_needs_header:
        csv_writer.writeheader()

    try:
        for i, preprint in enumerate(tqdm.tqdm(generator, total=total)):
            time.sleep(g.request_delay)

            item = preprint.get('item', {})
            pid = item.get('id')
            if not pid:
                continue

            if pid in existing_ids:
                continue

            # append JSONL line
            jsonl_f.write(json.dumps(preprint) + '\n')

            # extract CSV row
            license_name = (item.get('license') or {}).get('name')
            keywords_list = item.get('keywords') or []
            authors_list = item.get('authors') or []
            authors_names = []
            for a in authors_list:
                first = a.get('firstName') or ''
                last = a.get('lastName') or ''
                full = (first + ' ' + last).strip() or None
                if full:
                    authors_names.append(full)
            asset = item.get('asset') or {}
            pdf_url = (asset.get('original') or {}).get('url') if asset else None
            if not pdf_url and 'original' in asset:
                pdf_url = asset['original'].get('url')

            row = {
                'id': pid,
                'doi': item.get('doi'),
                'title': item.get('title'),
                'abstract': item.get('abstract'),
                'publishedDate': item.get('publishedDate'),
                'submittedDate': item.get('submittedDate'),
                'status': item.get('status'),
                'version': item.get('version'),
                'license': license_name,
                'keywords': '; '.join(keywords_list) if isinstance(keywords_list, list) else keywords_list,
                'authors': '; '.join(authors_names),
                'pdf_url': pdf_url,
            }
            csv_writer.writerow(row)
            num_written += 1
            existing_ids.add(pid)

            if num_written % g.batch_size == 0:
                csv_file.flush()
                jsonl_f.flush()
                time.sleep(g.batch_delay)

    finally:
        csv_file.close()
        jsonl_f.close()


def download_pdfs(cfg: ChemRxivConfig):
    d = cfg.download
    ensure_dir(d.download_dir)

    candidates = []
    with open(d.jsonl_path, 'r') as f_in:
        for line in f_in:
            try:
                preprint = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = preprint.get('item', {})
            pid = item.get('id')
            original = (item.get('asset') or {}).get('original') or {}
            url = original.get('url')
            if not pid or not url:
                continue
            filename = safe_filename(f"{pid}.pdf")
            filepath = os.path.join(d.download_dir, filename)
            if not os.path.exists(filepath):
                candidates.append((url, filepath))

    num = 0
    for url, filepath in tqdm.tqdm(candidates, total=len(candidates), desc="Downloading PDFs"):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                with open(filepath, 'wb') as out:
                    out.write(resp.content)
                num += 1
        except requests.exceptions.RequestException:
            time.sleep(d.cooldown_time)
        time.sleep(d.request_delay)

        if num > 0 and num % d.batch_size == 0:
            time.sleep(d.batch_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChemRxiv gatherer/downloader using YAML config with Pydantic")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stage", required=True,
                        choices=["gather", "download"], help="Stage to run")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        raw = yaml.safe_load(file)
    cfg = ChemRxivConfig(**raw)

    if args.stage == "gather":
        gather_metadata(cfg)
    elif args.stage == "download":
        download_pdfs(cfg)
