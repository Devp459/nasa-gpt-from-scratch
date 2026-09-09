"""
Download the actual document text for every link in links.txt and save
each as its own .txt file in OUTPUT_DIR, using a pool of concurrent
worker threads instead of downloading one file at a time. Most links are
NTRS "fulltext" URLs (already-extracted plain text from the PDF, no
parsing needed); a few might be raw "pdf" links if fulltext wasn't
available for that document — those get saved with a .pdf extension
instead so you can tell them apart later.

Resumable: skips urls already downloaded (tracked in a sibling
.done_urls file), so you can stop and re-run, or set LIMIT to do it in
batches.

Disclaimer: running this with a high MAX_WORKERS value, or against a
very large links.txt for an extended period, can send a large volume of
requests to NTRS in a short time and may trigger rate-limiting or
throttling on their end. Lower MAX_WORKERS if you see repeated errors,
and avoid running multiple instances of this script at once against the
same source.

Just edit the variables below and run: python pdf_downloader.py
"""


import re
import hashlib
import urllib.request
import urllib.error
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LINKS_FILE = "./links.txt"              # the file of URLs from fetch_ntrs_download_links.py
OUTPUT_DIR = "./data"                   # folder where downloaded text/pdf files get saved
LIMIT = None                            # max NEW urls to download this run (None = all remaining)
BASE_URL = "https://ntrs.nasa.gov"      # prefixed onto any link that's missing a scheme/host
MAX_WORKERS = 10                        # number of concurrent download threads

write_lock = threading.Lock()


def normalize_url(url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/"):
        url = "/" + url
    return BASE_URL + url


def filename_for_url(url):
    match = re.search(r"/citations/(\d+)/", url)
    if match:
        base = match.group(1)
    else:
        base = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = ".pdf" if url.lower().endswith(".pdf") or "/pdf" in url.lower() else ".txt"
    return base + ext


def load_done_urls(done_path):
    if not done_path.exists():
        return set()
    with open(done_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def download_one(url, out_dir):
    try:
        full_url = normalize_url(url)
        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()

        filename = filename_for_url(url)
        out_path = out_dir / filename
        if filename.endswith(".pdf"):
            with open(out_path, "wb") as out_f:
                out_f.write(content)
        else:
            with open(out_path, "w", encoding="utf-8", errors="replace") as out_f:
                out_f.write(content.decode("utf-8", errors="replace"))
        return url, None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return url, str(e)


def main():
    links_path = Path(LINKS_FILE)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_path = out_dir / ".done_urls"
    errors_path = out_dir / ".errors.log"

    done_urls = load_done_urls(done_path)
    print(f"{len(done_urls)} urls already downloaded previously — skipping those.")

    with open(links_path, "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    urls_to_fetch = [u for u in all_urls if u not in done_urls]
    if LIMIT is not None:
        urls_to_fetch = urls_to_fetch[:LIMIT]

    print(f"Downloading {len(urls_to_fetch)} file(s) this run with {MAX_WORKERS} workers...")

    done_out = open(done_path, "a", encoding="utf-8")
    err_out = open(errors_path, "a", encoding="utf-8")
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, url, out_dir): url for url in urls_to_fetch}
        for future in as_completed(futures):
            url, error = future.result()
            with write_lock:
                if error:
                    err_out.write(f"{url}\t{error}\n")
                    err_out.flush()
                done_out.write(url + "\n")
                done_out.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"...{completed}/{len(urls_to_fetch)} done")

    done_out.close()
    err_out.close()
    print("Done. Files saved to", out_dir.resolve())
    print("Errors (if any) logged to", errors_path.resolve())


if __name__ == "__main__":
    main()