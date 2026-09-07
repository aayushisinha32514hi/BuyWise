"""
BuyWise — Data Download Script
================================
Downloads the required datasets from Kaggle and places them in data/raw/.

Requirements:
    pip install kaggle
    Set up Kaggle API credentials:
        1. Go to https://www.kaggle.com/settings/account
        2. Click "Create New API Token" → downloads kaggle.json
        3. Place kaggle.json at ~/.kaggle/kaggle.json
        4. Run: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python3 data/download_data.py

See DATA_SOURCE.md for full dataset documentation.
"""

import os
import sys
import subprocess
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# ── Dataset identifiers ───────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "Smartprix Laptop Price Dataset 2024 (Primary)",
        "kaggle_id": "syedanwarafridi/laptop-price-dataset",
        "expected_file": "laptop_price.csv",
        "output_name": "smartprix_laptops_2024_raw.csv",
        "required": True,
    },
    # Uncomment below to also download the reviews dataset (for NLP component):
    # {
    #     "name": "Flipkart Laptop Reviews (NLP Supplement)",
    #     "kaggle_id": "TBD-verify-on-kaggle",  # search "laptop reviews flipkart" on Kaggle
    #     "expected_file": "TBD.csv",
    #     "output_name": "flipkart_laptop_reviews_raw.csv",
    #     "required": False,
    # },
]


def check_kaggle_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found.")
        print(f"Expected at: {kaggle_json}")
        print("\nSetup steps:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New API Token' → downloads kaggle.json")
        print("  3. Move kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)


def download_dataset(dataset: dict, output_dir: Path) -> bool:
    """Download a Kaggle dataset and rename to a consistent filename."""
    print(f"\n── Downloading: {dataset['name']} ──")
    print(f"   Kaggle ID: {dataset['kaggle_id']}")

    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                dataset["kaggle_id"],
                "--path", str(output_dir),
                "--unzip",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"   ERROR: {result.stderr.strip()}")
            return False

        # Rename to our consistent filename
        expected = output_dir / dataset["expected_file"]
        target = output_dir / dataset["output_name"]

        if expected.exists() and not target.exists():
            expected.rename(target)
            print(f"   Saved as: {target.name}")
        elif target.exists():
            print(f"   Already exists: {target.name}")
        else:
            # Kaggle may use a different filename — list what was downloaded
            downloaded = list(output_dir.glob("*.csv"))
            print(f"   WARNING: Expected file not found. Downloaded files: {[f.name for f in downloaded]}")
            print("   Please manually rename the file to:", dataset["output_name"])
            return False

        # Quick validation
        file_size_kb = target.stat().st_size / 1024
        print(f"   File size: {file_size_kb:.1f} KB")

        if file_size_kb < 10:
            print("   WARNING: File seems very small. Download may have failed.")
            return False

        return True

    except FileNotFoundError:
        print("   ERROR: 'kaggle' command not found.")
        print("   Install with: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("   ERROR: Download timed out.")
        return False


def main():
    print("BuyWise — Dataset Download")
    print("=" * 50)

    # Ensure raw data directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check credentials before attempting downloads
    check_kaggle_credentials()

    success_count = 0
    for dataset in DATASETS:
        ok = download_dataset(dataset, RAW_DIR)
        if ok:
            success_count += 1
        elif dataset["required"]:
            print(f"\nFATAL: Required dataset '{dataset['name']}' failed to download.")
            print("Cannot proceed with ML pipeline without this data.")
            sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Download complete: {success_count}/{len(DATASETS)} datasets ready")
    print(f"Raw data location: {RAW_DIR}")
    print("\nNext step: Run ml/scripts/preprocess.py to clean and prepare the data.")


if __name__ == "__main__":
    main()
