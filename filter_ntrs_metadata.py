"""
Stream-filter the huge NTRS bulk metadata JSON (ntrs-public-metadata.json,
~2GB) down to just the records that actually have full text available,
without loading the whole file into memory.

Output: an ndjson file (one JSON object per line) with the fields you need
to decide what to fetch next: id, title, stiType, center, subjectCategories,
distribution. Full abstract/authors are dropped here to keep the file small
and fast to page through — go back to the original file by id if you need
the rest of a record later.

NTRS bulk metadata link - https://sti.nasa.gov/harvesting-data-from-ntrs/

"""

import sys
import json
from pathlib import Path

READ_CHUNK = 1 << 20  # 1 MB per file read


def iter_top_level_records(path):
    """Yield (id, record_dict) for each entry in the outer JSON object,
    reading the file incrementally instead of json.load()-ing it whole."""

    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(READ_CHUNK)
        pos = buf.index("{") + 1

        def grow():
            """Read one more chunk and append it. Returns False only at true EOF."""
            nonlocal buf
            more = f.read(READ_CHUNK)
            if not more:
                return False
            buf += more
            return True

        def ensure(min_remaining):
            while len(buf) - pos < min_remaining:
                if not grow():
                    return False
            return True

        def parse_at(pos):
            """Try to parse a JSON value starting at pos, growing the buffer
            one chunk at a time until it parses or we truly hit EOF."""
            nonlocal buf
            while True:
                try:
                    return decoder.raw_decode(buf, pos)
                except ValueError:
                    if not grow():
                        raise

        while True:
            # bound memory drop what we've already consumed
            if pos > READ_CHUNK:
                buf = buf[pos:]
                pos = 0

            ensure(4)
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
                if pos >= len(buf):
                    ensure(4)
            if pos >= len(buf) or buf[pos] == "}":
                break

            # parse the key its a JSON string
            try:
                key, pos = parse_at(pos)
            except ValueError:
                return

            ensure(4)
            while pos < len(buf) and buf[pos] in " \t\r\n":
                pos += 1
                ensure(4)
            pos += 1  # the colon
            ensure(4)
            while pos < len(buf) and buf[pos] in " \t\r\n":
                pos += 1
                ensure(4)

            # parse the value its a JSON object
            try:
                value, pos = parse_at(pos)
            except ValueError:
                return

            yield key, value


def main():
    input_path = Path("./ntrs-public-metadata.json/ntrs-public-metadata.json")        # path to the original NTRS bulk metadata JSON file
    output_path = Path("./filtered.ndjson")                                            # path to the output filtered ndjson file

    seen = 0
    kept = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for doc_id, record in iter_top_level_records(input_path):
            seen += 1
            if seen % 50000 == 0:
                print(f"...scanned {seen} records, kept {kept} so far")

            # Only keep records where NASA actually disseminated the full
            # document not just a citation/abstract.
            if record.get("disseminated") != "DOCUMENT_AND_METADATA":
                continue

            center = record.get("center") or {}
            slim = {
                "id": doc_id,
                "title": record.get("title", ""),
                "stiType": record.get("stiType", ""),
                "center_code": center.get("code", ""),
                "center_name": center.get("name", ""),
                "subjectCategories": record.get("subjectCategories", []),
                "distribution": record.get("distribution", ""),
            }
            out.write(json.dumps(slim, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. Scanned {seen} total records, kept {kept} with full text available.")
    print(f"Written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()