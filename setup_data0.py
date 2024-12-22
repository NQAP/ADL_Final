"""Download text-to-SQL datasets and unzip them."""

import logging
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, save_path: str) -> None:
    # Stream the download in chunks
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    chunk_size = 1024  # 1 KB

    with (
        open(save_path, "wb") as f,
        tqdm(
            desc=save_path,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                bar.update(len(chunk))


def download_bird(save_dir: str) -> None:
    logger.info("Downloading and unzipping BIRD...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Download BIRD
    bird_link = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    bird_zip = "bird.zip"
    bird_save_path = f"{save_dir}/{bird_zip}"
    if os.path.exists(bird_save_path):
        logger.info("BIRD already exists at %s", bird_save_path)
    else:
        logger.info("Downloading BIRD to %s", bird_save_path)
        download_file(url=bird_link, save_path=bird_save_path)

    # Unzip BIRD
    extract_to_dir = os.path.join(save_dir, "bird")
    if os.path.exists(extract_to_dir):
        logger.info("BIRD already unzipped to %s", extract_to_dir)

    else:
        with zipfile.ZipFile(bird_save_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)

        # Search for the file path of "dev_databases.zip" in the folders
        for root, _, files in os.walk(save_dir):
            for file in files:
                if file == "dev_databases.zip":
                    db_zip_path = os.path.join(root, file)
                    break

        with zipfile.ZipFile(db_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_dir)
        logger.info("Unzipped to %s", extract_to_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_dir = "./data"
    download_bird(save_dir)
