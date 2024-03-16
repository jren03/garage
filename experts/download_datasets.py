"""
This file downloads the exact datasets used in the paper.
They will be downloaded to experts/<env_name>/<env_name>_demos.npz.
NOTE: for antmaze datasets, please run
    python experts/collect_demos.py --env antmaze-large-diverse-v2
    or
    python experts/collect_demos.py --env antmaze-large-play-v2
to generate the demonstration files.

Users may specify to download a specific environment by passing the --env flag,
or all environments with the --all flag.
"""

import argparse
import parser
import wget
from pathlib import Path

from garage.utils.common import (
    PROJECT_ROOT,
    ENV_ABBRV_TO_FULL,
    ENV_ABBRV_TO_DATASET_URL,
)
from garage.utils.fetch_demos import fetch_demos


def download_demos(env, download_all_envs=False):
    if download_all_envs:
        download_urls = [
            (ENV_ABBRV_TO_FULL[env], ENV_ABBRV_TO_DATASET_URL[env])
            for env in ENV_ABBRV_TO_DATASET_URL.keys()
        ]
    else:
        download_urls = [(ENV_ABBRV_TO_FULL[env], ENV_ABBRV_TO_DATASET_URL[env])]

    for env, url in download_urls:
        print(f"Downloading {env} demonstrations...")
        dataset_path = Path(PROJECT_ROOT / f"experts/{env}/{env}_demos.npz")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        wget.download(url, str(dataset_path))
        print(f"\nDownloaded to {dataset_path}, verifying dataset contents...")
        fetch_demos(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download expert demonstrations.")
    parser.add_argument(
        "--env", choices=["ant", "hopper", "humanoid", "walker"], default="ant"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all expert demonstrations.",
    )
    args = parser.parse_args()

    download_demos(args.env, args.all)
