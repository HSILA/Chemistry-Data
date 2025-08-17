from pydantic import BaseModel, Field
from typing import Optional, List


class DownloadConfig(BaseModel):
    save_path: str = Field(..., description="Directory to save downloaded JSON files")
    max_cid: int = Field(..., description="Maximum compound CID to attempt when continuing from last")
    batch_size: int = Field(1000, description="Number of successful requests per batch before pausing")
    batch_delay: int = Field(120, description="Pause in seconds after each batch completes")
    cooldown_time: int = Field(120, description="Cooldown in seconds after connection errors before retrying loop")
    request_delay: int = Field(3, description="Delay in seconds between requests to PubChem")
    predefined_cids_path: Optional[str] = Field(
        default=None,
        description="Path to a JSON file of CIDs to retrieve. If provided, downloader only retrieves these CIDs.",
    )


class ParseConfig(BaseModel):
    jsons_dir: str = Field("./PubChem", description="Directory containing downloaded JSON files")
    comp_csv: str = Field("compounds.csv", description="Path to write compounds table")
    desc_csv: str = Field("descriptions.csv", description="Path to write descriptions table")
    batch_size: int = Field(1000, description="Batch size for writing rows to CSV")


class PubChemConfig(BaseModel):
    download: DownloadConfig = Field(..., description="Download configuration")
    parse: ParseConfig = Field(..., description="Parsing configuration")


