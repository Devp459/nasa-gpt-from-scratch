"""
Given the filtered.ndjson produced by filter_ntrs_metadata.py, hit the
live NTRS API per document ID to get its real download links, and write
the full-text links to links.txt.

For each id this calls: https://ntrs.nasa.gov/api/citations/{id}/downloads
which returns a JSON list of available files, each with a "links" object
containing "pdf" and "fulltext" URLs. "fulltext" returns already-extracted
plain text (no PDF parsing needed), so that's what gets saved by default.

This is resumable: it skips ids it already has a link for (tracked in a
sibling .done_ids file), so you can stop it and re-run later, or set
LIMIT to do it in batches instead of all ~310k at once.

Just edit the variables below and run: python fetch_ntrs_download_links.py
"""


import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

CANDIDATES_FILE = "./filtered.ndjson"         # filtered file path from filter_ntrs_metadata.py
OUTPUT_FILE = "./links.txt"                   # output file path
CHUNK_SIZE = 5000


def extract_link(line):
    line = line.strip()
    if not line:
        return None
    try:
        doc_id = str(json.loads(line)["id"])
        return f"/api/citations/{doc_id}/downloads/{doc_id}.txt"
    except Exception:
        return None


def main():
    candidates_path = Path(CANDIDATES_FILE)
    output_path = Path(OUTPUT_FILE)

    with open(candidates_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with Pool(cpu_count()) as pool, open(output_path, "w", encoding="utf-8") as out:
        count = 0
        for link in pool.imap(extract_link, lines, chunksize=CHUNK_SIZE):
            if link is not None:
                out.write(link + "\n")
                count += 1

    print(count, "links written to", output_path.resolve())


if __name__ == "__main__":
    main()