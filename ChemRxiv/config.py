from pydantic import BaseModel, Field


class GatherConfig(BaseModel):
    jsonl_path: str = Field(
        "./ChemRxiv/chemrxiv_metadata.jsonl",
        description="Path to append JSONL metadata lines",
    )
    csv_path: str = Field(
        "./ChemRxiv/chemrxiv_metadata.csv",
        description="Path to write CSV with ChemRxiv metadata",
    )
    batch_size: int = Field(
        1000, description="Number of gathered items per batch before pausing"
    )
    batch_delay: int = Field(
        120, description="Pause in seconds after each batch completes"
    )
    cooldown_time: int = Field(
        120,
        description="Cooldown in seconds after connection errors before retrying loop",
    )
    request_delay: float = Field(
        1.0, description="Delay in seconds between requests to ChemRxiv API"
    )


class DownloadConfig(BaseModel):
    jsonl_path: str = Field(
        "./ChemRxiv/chemrxiv_metadata.jsonl",
        description="Path to JSONL file created in gather stage",
    )
    download_dir: str = Field(
        "./ChemRxiv/papers", description="Directory to save downloaded PDF files"
    )
    batch_size: int = Field(
        1000, description="Number of successful downloads per batch before pausing"
    )
    batch_delay: int = Field(
        120, description="Pause in seconds after each batch completes"
    )
    cooldown_time: int = Field(
        120,
        description="Cooldown in seconds after connection errors before retrying loop",
    )
    request_delay: float = Field(
        1.0, description="Delay in seconds between consecutive downloads"
    )


class ChemRxivConfig(BaseModel):
    gather: GatherConfig = Field(..., description="Gather stage configuration")
    download: DownloadConfig = Field(..., description="Download stage configuration")
