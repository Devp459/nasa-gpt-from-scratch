"""
Download the actual document text for every link in links.txt and save
each as its own .txt file in OUTPUT_DIR. Most links are NTRS "fulltext"
URLs (already-extracted plain text from the PDF, no parsing needed);
a few might be raw "pdf" links if fulltext wasn't available
for that document — those get saved with a .pdf extension instead so
you can tell them apart later.

Resumable: skips urls already downloaded (tracked in a sibling
.done_urls file), so you can stop and re-run, or set LIMIT to do it in
batches.

Just edit the variables below and run: python download_ntrs_text.py
"""

import json
import time
import re
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

# ---- fill these in ----
LINKS_FILE = "./links.txt"              # the file of URLs from fetch_ntrs_download_links.py
OUTPUT_DIR = "./data"                   # folder where downloaded text/pdf files get saved
LIMIT = None                            # max NEW urls to download this run (None = all remaining)
DELAY_SECONDS = 0.3                     # pause between downloads, be polite
BASE_URL = "https://ntrs.nasa.gov"      # prefixed onto any link that's missing a scheme/host
# ------------------------


def normalize_url(url: str) -> str:
    """The NTRS downloads API returns relative paths like
    '/api/citations/123/downloads/123.txt' — turn those into full URLs."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/"):
        url = "/" + url
    return BASE_URL + url


def filename_for_url(url: str) -> str:
    """Pull the NTRS citation id out of the URL if present, else hash the url."""
    match = re.search(r"/citations/(\d+)/", url)
    if match:
        base = match.group(1)
    else:
        base = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

    ext = ".pdf" if url.lower().endswith(".pdf") or "/pdf" in url.lower() else ".txt"
    return base + ext


def load_done_urls(done_path: Path):
    if not done_path.exists():
        return set()
    with open(done_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


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

    print(f"Downloading {len(urls_to_fetch)} file(s) this run...")

    with open(done_path, "a", encoding="utf-8") as done_out, \
         open(errors_path, "a", encoding="utf-8") as err_out:

        for i, url in enumerate(urls_to_fetch, 1):
            try:
                full_url = normalize_url(url)
                req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read()

                filename = filename_for_url(url)
                out_path = out_dir / filename
                mode = "wb" if filename.endswith(".pdf") else "w"
                if mode == "wb":
                    with open(out_path, "wb") as out_f:
                        out_f.write(content)
                else:
                    with open(out_path, "w", encoding="utf-8", errors="replace") as out_f:
                        out_f.write(content.decode("utf-8", errors="replace"))

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                err_out.write(f"{url}\t{e}\n")
                err_out.flush()

            # record the ORIGINAL url (as it appears in links.txt) so resuming
            # still matches correctly against that file
            done_out.write(url + "\n")
            done_out.flush()

            if i % 100 == 0:
                print(f"...{i}/{len(urls_to_fetch)} done")

            time.sleep(DELAY_SECONDS)

    print("Done. Files saved to", out_dir.resolve())
    print("Errors (if any) logged to", errors_path.resolve())


if __name__ == "__main__":
    main()
