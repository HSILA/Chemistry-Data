import requests
import json
import time
import os
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import tqdm

SAVE_PATH = "./PubChem"
MAX = 50000
BATCH_SIZE = 1000
BATCH_DELAY = 120
COOLDOWN_TIME = 120
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}

os.makedirs(SAVE_PATH, exist_ok=True)


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=64),
    stop=stop_after_attempt(5),
)
def robust_get(url):
    return requests.get(url, headers=headers, timeout=10)


def get_last_cid(save_path):
    files = [
        f for f in os.listdir(save_path) if f.startswith("cid_") and f.endswith(".json")
    ]
    if not files:
        return 0
    cids = [int(f.split("_")[1].split(".")[0]) for f in files]
    return max(cids)


last_cid = get_last_cid(SAVE_PATH)
message = (
    "Previous download not found. Starting from scratch."
    if last_cid == 0
    else f"Start downloading from CID: {last_cid}"
)
print(message)
start_cid = last_cid + 1
num_requests = 0


for cid in tqdm.tqdm(range(start_cid, MAX + 1), total=MAX, initial=start_cid):
    while True:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
        )

        retry_count = 0

        try:
            response = robust_get(url)
            if response.status_code == 200:
                data = response.json()
                with open(f"{SAVE_PATH}/cid_{cid}.json", "w") as f:
                    json.dump(data, f)
                num_requests += 1
            break

        except requests.ConnectionError as e:
            print(
                f"Failed to fetch CID {cid} after retries: {e}. Waiting {COOLDOWN_TIME}s before continuing."
            )
            time.sleep(COOLDOWN_TIME)
        except Exception as e:
            print(f"Unexpected error for CID {cid}: {e}. Skipping.")
            break

    time.sleep(3)

    if num_requests > 0 and num_requests % BATCH_SIZE == 0:
        print(
            f"\nBatch complete: {num_requests} requests sent. Waiting {BATCH_DELAY}s..."
        )
        time.sleep(BATCH_DELAY)
