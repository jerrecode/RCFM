#!/usr/bin/env python3
import os
import json
import time
import requests
from tqdm import tqdm
import datetime

# Base directory is the folder containing this script (matches your PWD)
DATA_DIR = Path(__file__).resolve().parents[3] / "Data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
PROGRESS_PATH = os.path.join(DATA_DIR, ".progress")
LOG_PATH = os.path.join(DATA_DIR, "download_log.txt")

# Common headers to prevent 403/404 on some servers (GitHub, SDSS, IRSA, etc.)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)

def get_file_size(url):
    """HEAD request to pre-calculate size (with User-Agent for better compatibility)."""
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=15)
        if r.status_code == 200:
            return int(r.headers.get("content-length", 0))
        return 0
    except Exception:
        return 0

def main():
    manifest = load_manifest()

    # Pre-calculate total size for overall progress bar
    print("Pre-calculating total download size...")
    total_size = 0
    for entry in manifest:
        size = get_file_size(entry["url"])
        total_size += size
    print(f"Total estimated size: {total_size / (1024*1024):.2f} MB\n")

    # Load or create progress file
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            progress = json.load(f)
        completed = progress.get("completed", [])
        current_section = progress.get("current_section")
        start_time = progress.get("start_time", time.time())
        print("Resuming from previous run (partial downloads will be redone from scratch).\n")
    else:
        progress = {
            "completed": [],
            "current_section": None,
            "start_time": time.time(),
            "total_size": total_size
        }
        completed = []
        current_section = None
        start_time = progress["start_time"]
        with open(PROGRESS_PATH, "w") as f:
            json.dump(progress, f, indent=2)

    # If previous run crashed mid-file, delete the partial .tmp to avoid corruption
    if current_section:
        tmp_path = os.path.join(DATA_DIR, current_section + ".tmp")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Cleaned up partial file for '{current_section}' - will redo from start.")
        progress["current_section"] = None
        with open(PROGRESS_PATH, "w") as f:
            json.dump(progress, f, indent=2)

    # Overall progress bar (bytes)
    overall_bar = tqdm(
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Overall",
        initial=0,
        position=0
    )

    failed = []          # list of (name, url, relpath, error_msg)
    errors = 0

    for entry in manifest:
        name = entry["name"]
        url = entry["url"]
        relpath = entry["relpath"]
        full_path = os.path.join(DATA_DIR, relpath)

        # Create directory structure
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Skip if already completed (in progress file or file exists and is non-zero)
        if relpath in completed or (os.path.exists(full_path) and os.path.getsize(full_path) > 0):
            size = os.path.getsize(full_path) if os.path.exists(full_path) else get_file_size(url)
            overall_bar.update(size)
            continue

        # Mark as current (for resume safety)
        progress["current_section"] = relpath
        with open(PROGRESS_PATH, "w") as f:
            json.dump(progress, f, indent=2)

        tmp_path = full_path + ".tmp"
        print(f"\nDownloading: {name} → {relpath}")

        try:
            r = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            r.raise_for_status()
            file_size = int(r.headers.get("content-length", 0))

            with open(tmp_path, "wb") as f:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=name[:40],
                    leave=False,
                    position=1
                ) as file_bar:
                    for chunk in r.iter_content(chunk_size=8192 * 4):
                        if chunk:
                            f.write(chunk)
                            file_bar.update(len(chunk))
                            overall_bar.update(len(chunk))

        except Exception as e:
            errors += 1
            error_msg = str(e)
            print(f"ERROR downloading {name}: {error_msg}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # Log failure with full details
            with open(LOG_PATH, "a") as logf:
                logf.write(f"{datetime.datetime.now().isoformat()} | FAILED {name} | reason: {error_msg} | url: {url} | relpath: {relpath}\n")

            failed.append((name, url, relpath, error_msg))
            progress["current_section"] = None
            with open(PROGRESS_PATH, "w") as f:
                json.dump(progress, f, indent=2)
            continue

        # Success: move tmp → final
        os.rename(tmp_path, full_path)

        # Mark completed
        completed.append(relpath)
        progress["completed"] = completed
        progress["current_section"] = None
        with open(PROGRESS_PATH, "w") as f:
            json.dump(progress, f, indent=2)

        # Log success
        with open(LOG_PATH, "a") as logf:
            logf.write(f"{datetime.datetime.now().isoformat()} | Completed {name} ({file_size / (1024*1024):.2f} MB)\n")

        # Estimate remaining time / average speed
        elapsed_total = time.time() - start_time
        done_bytes = overall_bar.n
        if elapsed_total > 0 and done_bytes > 0:
            speed_mb_s = done_bytes / elapsed_total / (1024 * 1024)
            remaining_bytes = total_size - done_bytes
            if remaining_bytes > 0 and speed_mb_s > 0:
                eta_seconds = remaining_bytes / (speed_mb_s * 1024 * 1024)
                print(f"   Avg speed: {speed_mb_s:.2f} MB/s | ETA: {eta_seconds/60:.1f} min | Time elapsed: {elapsed_total/60:.1f} min")
            else:
                print(f"   Avg speed: {speed_mb_s:.2f} MB/s | Time elapsed: {elapsed_total/60:.1f} min")

    overall_bar.close()

    # ==================== FINAL DOWNLOAD REPORT ====================
    print("\n" + "=" * 70)
    print("DOWNLOAD REPORT")
    print("=" * 70)
    print(f"Total estimated size       : {total_size / (1024*1024):.2f} MB")
    print(f"Successfully completed     : {len(completed)} files")
    print(f"Failed / skipped due to error : {len(failed)} files")
    print(f"Total runtime              : {(time.time() - start_time)/60:.1f} minutes")

    if failed:
        print("\nFailed downloads:")
        for name, url, relpath, reason in failed:
            print(f"  • {name}")
            print(f"    Path : {relpath}")
            print(f"    URL  : {url}")
            print(f"    Error: {reason}")
            print()
    else:
        print("\n✅ All datasets downloaded successfully!")

    print("Check raw/ folder and download_log.txt for full details.")
    print("=" * 70)

if __name__ == "__main__":
    main()
