"""Data ingestion script for UCI Individual Household Electric Power Consumption dataset."""
import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URLS: List[str] = [
    "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
    "https://d396qusza40orc.cloudfront.net/exdata%2Fdata%2Fhousehold_power_consumption.zip",
]


def download_and_extract(
    dest_path: Path,
    urls: Optional[List[str]] = None,
) -> Path:
    """Download and extract the dataset to dest_path if not already present.

    Args:
        dest_path: Target path for the extracted txt file.
        urls: List of fallback download URLs.

    Returns:
        Path to the extracted data file.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Raw dataset file already exists at %s. Skipping download.", dest_path)
        return dest_path

    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "household_power_consumption.zip"

    candidate_urls = urls or DEFAULT_URLS
    download_success = False

    for url in candidate_urls:
        logger.info("Attempting download from: %s", url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_f:
                out_f.write(response.read())

            if zip_path.exists() and zip_path.stat().st_size > 1000:
                logger.info("Downloaded %d bytes to %s", zip_path.stat().st_size, zip_path)
                download_success = True
                break
        except Exception as exc:
            logger.warning("Download from %s failed: %s", url, exc)
            if zip_path.exists():
                zip_path.unlink()

    if not download_success:
        raise RuntimeError(
            f"Failed to download dataset from all candidate URLs: {candidate_urls}. "
            f"Please place household_power_consumption.txt manually at {dest_path}"
        )

    logger.info("Extracting %s into %s...", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Clean up zip archive to save disk space
    if zip_path.exists():
        zip_path.unlink()

    if not dest_path.exists():
        # Search if extracted with different name or in subfolder
        found = list(dest_dir.glob("*.txt"))
        if found:
            logger.info("Found extracted text file: %s", found[0])
            return found[0]
        raise FileNotFoundError(f"Extraction completed but {dest_path} not found.")

    logger.info("Raw dataset ready at: %s", dest_path)
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UCI Household Power Consumption dataset")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params_path = Path(args.params)
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        raw_path = Path(params.get("dataset", {}).get("raw_path", "data/raw/household_power_consumption.txt"))
        uci_url = params.get("dataset", {}).get("uci_url")
        urls = [uci_url] + DEFAULT_URLS if uci_url else DEFAULT_URLS
    else:
        raw_path = Path("data/raw/household_power_consumption.txt")
        urls = DEFAULT_URLS

    download_and_extract(dest_path=raw_path, urls=urls)


if __name__ == "__main__":
    main()

