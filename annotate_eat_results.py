#!/usr/bin/env python3
"""
Annotate an existing eat_result.txt with UniProt information for each Lookup-ID.

eat.py writes "None" in the Lookup-Label column when no --lookupLabels file is
passed. This script fills that column post hoc by querying the UniProt REST API
for the Lookup-IDs actually present in the result file (no need to re-run EAT).

Accessions that are no longer primary in UniProt (merged/demerged since the
lookup DB snapshot) are retried as secondary accessions; anything still not
found is reported as "NA".

Only the Python standard library is used.

Example:
    python annotate_eat_results.py -i results/eat_result.txt
    python annotate_eat_results.py -i eat_result.txt -o annotated.tsv \
        --fields protein_name,organism_name,gene_names,ec
"""
import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UNIPROT = "https://rest.uniprot.org/uniprotkb"
CHUNK_SIZE = 500  # UniProt accepts up to ~1000 accessions per /accessions call
# official UniProtKB accession format; anything else makes the API reject the whole request
ACC_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")


def fetch_tsv(url, params, retries=3):
    """GET a UniProt TSV endpoint and return rows as a list of dicts."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(full_url, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            return list(csv.DictReader(text.splitlines(), delimiter="\t"))
        except (urllib.error.URLError, TimeoutError) as e:
            # client errors (4xx) will not succeed on retry
            is_client_err = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500
            if attempt == retries or is_client_err:
                raise
            print("UniProt request failed ({}), retrying...".format(e), file=sys.stderr)
            time.sleep(5 * attempt)


def query_uniprot(accessions, fields):
    """Return {accession: {column_name: value}} for the given accessions."""
    field_str = "accession," + ",".join(fields)
    annotations = dict()
    header = None

    invalid = [acc for acc in accessions if not ACC_RE.match(acc)]
    if invalid:
        print("WARNING: skipping {} IDs that are not UniProt accessions: {}".format(
            len(invalid), ", ".join(invalid)), file=sys.stderr)
    accessions = [acc for acc in accessions if ACC_RE.match(acc)]

    # primary accessions, in chunks
    for i in range(0, len(accessions), CHUNK_SIZE):
        chunk = accessions[i:i + CHUNK_SIZE]
        rows = fetch_tsv(UNIPROT + "/accessions",
                         {"accessions": ",".join(chunk), "format": "tsv", "fields": field_str})
        for row in rows:
            header = header or [k for k in row if k != "Entry"]
            annotations[row["Entry"]] = {k: v for k, v in row.items() if k != "Entry"}

    # fall back to secondary accessions for anything not found
    missing = [acc for acc in accessions if acc not in annotations]
    for acc in missing:
        rows = fetch_tsv(UNIPROT + "/search",
                         {"query": "sec_acc:" + acc, "format": "tsv", "fields": field_str, "size": 10})
        if rows:
            # a demerged accession maps to several current entries: annotate with the
            # first one and list all of them in Current-Accession
            header = header or [k for k in rows[0] if k != "Entry"]
            annotations[acc] = {k: v for k, v in rows[0].items() if k != "Entry"}
            annotations[acc]["Current-Accession"] = ";".join(row["Entry"] for row in rows)

    return annotations, header or []


def create_arg_parser():
    parser = argparse.ArgumentParser(description=(
        "Fill the Lookup-Label column of an EAT result file (eat_result.txt) "
        "with UniProt annotations of the Lookup-IDs."))
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to eat_result.txt produced by eat.py.")
    parser.add_argument("-o", "--output", required=False, type=str, default=None,
                        help="Output TSV. Default: <input>_annotated.tsv next to the input.")
    parser.add_argument("--fields", type=str, default="protein_name,organism_name,gene_names",
                        help="Comma-separated UniProt return fields (see "
                             "https://rest.uniprot.org/configure/uniprotkb/result-fields). "
                             "The first field fills Lookup-Label; the others are appended as columns. "
                             "Default: protein_name,organism_name,gene_names")
    return parser


def main():
    args = create_arg_parser().parse_args()
    in_p = Path(args.input)
    out_p = Path(args.output) if args.output else in_p.with_name(in_p.stem + "_annotated.tsv")
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    with open(in_p, "r") as in_f:
        reader = csv.DictReader(in_f, delimiter="\t")
        in_header = reader.fieldnames
        results = list(reader)

    if "Lookup-ID" not in in_header or "Lookup-Label" not in in_header:
        sys.exit("Input does not look like an EAT result file (missing Lookup-ID/Lookup-Label).")

    accessions = sorted({row["Lookup-ID"] for row in results})
    print("Querying UniProt for {} unique Lookup-IDs...".format(len(accessions)))
    annotations, uniprot_cols = query_uniprot(accessions, fields)

    label_col, extra_cols = (uniprot_cols[0], uniprot_cols[1:]) if uniprot_cols else (None, [])
    out_header = in_header + extra_cols + ["Current-Accession"]

    with open(out_p, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_header, delimiter="\t")
        writer.writeheader()
        for row in results:
            ann = annotations.get(row["Lookup-ID"])
            if ann is None:
                ann = dict()
            row["Lookup-Label"] = ann.get(label_col, "NA") or "NA"
            for col in extra_cols:
                row[col] = ann.get(col, "NA") or "NA"
            row["Current-Accession"] = ann.get("Current-Accession", row["Lookup-ID"] if ann else "NA")
            writer.writerow(row)

    not_found = [acc for acc in accessions if acc not in annotations]
    if not_found:
        print("WARNING: {} Lookup-IDs not found in UniProt (written as NA): {}".format(
            len(not_found), ", ".join(not_found)), file=sys.stderr)
    print("Wrote {} rows to {}".format(len(results), out_p))


if __name__ == "__main__":
    main()
