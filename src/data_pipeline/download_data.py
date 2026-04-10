"""
download_data.py - Fetch raw data and trained models from GitHub Releases.

After cloning the repo, teammates run:

    python src/data_pipeline/download_data.py                          # raw data
    python src/data_pipeline/download_data.py --tag v0.2-trained-models  # models
    python src/data_pipeline/download_data.py --all                    # both

All assets are downloaded to the correct local directories based on
filename patterns. Files already present (with matching size) are skipped.

Requires: Python 3.8+ stdlib only (no pip dependencies).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration — update these after creating the GitHub repo
# ---------------------------------------------------------------------------
GITHUB_OWNER = "aniruddha1295"  # GitHub org or username
GITHUB_REPO = "CostalCred"  # Repository name
DEFAULT_TAG = "v0.1-raw-data"
MODEL_TAG = "v0.2-trained-models"

# Project root is three levels up from this file:
#   src/data_pipeline/download_data.py -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Asset-to-directory mapping (by filename pattern, checked in order)
# ---------------------------------------------------------------------------
# Each entry is (substring_or_suffix, target_directory_relative_to_root).
# The first match wins, so put more specific patterns before generic ones.
ASSET_ROUTING = [
    # Sentinel-2 composites (site_year.tif)
    ("sundarbans_", "data/raw/sentinel2"),
    ("gulf_of_kutch_", "data/raw/sentinel2"),
    ("pichavaram_", "data/raw/sentinel2"),
    # GMW vector data
    (".gpkg", "data/raw/gmw"),
    (".shp", "data/raw/gmw"),
    (".shx", "data/raw/gmw"),
    (".dbf", "data/raw/gmw"),
    (".prj", "data/raw/gmw"),
    (".cpg", "data/raw/gmw"),
    ("gmw", "data/raw/gmw"),
    # Trained model weights
    (".pt", "models"),
    (".pth", "models"),
    ("xgboost_model", "models"),
    # Fallback for any remaining .tif files
    (".tif", "data/raw/sentinel2"),
    (".tiff", "data/raw/sentinel2"),
]

# Files with these extensions are never routed to models/ via the generic
# .json rule — avoids grabbing metric result JSONs by accident.
MODEL_JSON_PREFIXES = ("xgboost_model",)

GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_url(tag: str) -> str:
    """Return the GitHub API URL for a release by tag."""
    return f"{GITHUB_API}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tags/{tag}"


def _format_bytes(n: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _resolve_dir(filename: str) -> str | None:
    """Return the local directory (relative to project root) for a filename."""
    lower = filename.lower()
    for pattern, directory in ASSET_ROUTING:
        if pattern in lower:
            return directory
    return None


def fetch_release_assets(tag: str) -> list[dict]:
    """Query the GitHub API for release assets under *tag*.

    Returns a list of dicts with keys: name, size, browser_download_url.
    """
    url = _api_url(tag)
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"\nERROR: Release '{tag}' not found.")
            print(f"  URL checked: {url}")
            print(
                f"\nMake sure the release exists at:\n"
                f"  https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tag/{tag}\n"
            )
            print(
                "If the repository has moved, update GITHUB_OWNER / GITHUB_REPO at the "
                "top of this file."
            )
            sys.exit(1)
        raise
    except urllib.error.URLError as exc:
        print(f"\nERROR: Could not reach GitHub API — {exc.reason}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    assets = data.get("assets", [])
    if not assets:
        print(f"\nWARNING: Release '{tag}' exists but has no assets attached.")
    return assets


def download_file(url: str, dest: str, expected_size: int) -> None:
    """Download *url* to *dest*, showing a progress bar."""
    tmp = dest + ".part"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fout:
            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB
            start = time.monotonic()
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, expected_size, start)
    except Exception:
        # Clean up partial file on failure
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    os.replace(tmp, dest)
    print()  # newline after progress bar


def _print_progress(downloaded: int, total: int, start: float) -> None:
    """Print an in-place progress line."""
    elapsed = time.monotonic() - start
    pct = downloaded / total * 100 if total else 0
    speed = downloaded / elapsed if elapsed > 0 else 0
    bar_len = 30
    filled = int(bar_len * downloaded / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r  [{bar}] {pct:5.1f}%  {_format_bytes(downloaded)}/{_format_bytes(total)}"
        f"  {_format_bytes(speed)}/s"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def download_release(tag: str, *, dry_run: bool = False) -> tuple[int, int]:
    """Download all assets from a single release tag.

    Returns (downloaded_count, skipped_count).
    """
    print(f"\n{'=' * 60}")
    print(f"  Release: {tag}")
    print(f"  Repo:    {GITHUB_OWNER}/{GITHUB_REPO}")
    print(f"{'=' * 60}\n")

    assets = fetch_release_assets(tag)
    if not assets:
        return 0, 0

    downloaded = 0
    skipped = 0
    total_bytes = 0

    for asset in assets:
        name = asset["name"]
        size = asset["size"]
        url = asset["browser_download_url"]

        target_dir = _resolve_dir(name)
        if target_dir is None:
            print(f"  SKIP (unknown type): {name}")
            skipped += 1
            continue

        dest_dir = os.path.join(PROJECT_ROOT, target_dir)
        dest_path = os.path.join(dest_dir, name)

        # Skip if already downloaded and size matches
        if os.path.exists(dest_path):
            existing_size = os.path.getsize(dest_path)
            if existing_size == size:
                print(f"  SKIP (exists, {_format_bytes(size)}): {name}")
                skipped += 1
                continue
            print(
                f"  RE-DOWNLOAD (size mismatch: local {_format_bytes(existing_size)} "
                f"vs remote {_format_bytes(size)}): {name}"
            )

        if dry_run:
            print(f"  WOULD DOWNLOAD ({_format_bytes(size)}): {name} -> {target_dir}/")
            continue

        os.makedirs(dest_dir, exist_ok=True)
        print(f"  Downloading {name} ({_format_bytes(size)}) -> {target_dir}/")
        download_file(url, dest_path, size)
        downloaded += 1
        total_bytes += size

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped.")
    if total_bytes:
        print(f"Total downloaded: {_format_bytes(total_bytes)}")
    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CoastalCred data from GitHub Releases.",
        epilog=(
            "Examples:\n"
            "  python download_data.py                          # raw data\n"
            "  python download_data.py --tag v0.2-trained-models  # models\n"
            "  python download_data.py --all                    # both\n"
            "  python download_data.py --dry-run                # preview only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Release tag to download (default: {DEFAULT_TAG})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help=f"Download both {DEFAULT_TAG} and {MODEL_TAG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    tags = [DEFAULT_TAG, MODEL_TAG] if args.download_all else [args.tag]

    grand_downloaded = 0
    grand_skipped = 0

    for tag in tags:
        d, s = download_release(tag, dry_run=args.dry_run)
        grand_downloaded += d
        grand_skipped += s

    if len(tags) > 1:
        print(f"\n{'=' * 60}")
        print(f"  TOTAL: {grand_downloaded} downloaded, {grand_skipped} skipped")
        print(f"{'=' * 60}")

    if grand_downloaded == 0 and grand_skipped == 0:
        print(
            "\nNo assets found. Have you published the release(s) yet?\n"
            "See CLAUDE.md section 'Data sharing strategy' for instructions."
        )


if __name__ == "__main__":
    main()
