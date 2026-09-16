#!/usr/bin/env python3
"""
Takes a radiomics CSV (e.g. radiomic_features.csv or features.csv) and
generates an Excel (.xlsx) file where each sheet contains the columns of a
single feature type: shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm,
diagnostics...

The "type" is detected from the column name, which in pyradiomics follows
the pattern:

    <filter>_<type>_<featureName>          e.g. original_shape_Elongation
    <filter>_<type>_<featureName>-MODALITY e.g. wavelet-LLL_glcm_Idn-PET

and for diagnostics columns:

    diagnostics_<Category>-<filter>_<name> e.g. diagnostics_Image-original_Mean

Runs with no arguments, prompting interactively for:
  1. The path to the input CSV.
  2. The identifier column (default "Patient"), which is repeated on every
     sheet so the data can be cross-referenced later.
  3. The path to the output .xlsx file (default: same name as the CSV).

Requires pandas and openpyxl (if you don't have them: pip install pandas openpyxl).
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd

# Feature types known in pyradiomics (in addition to "diagnostics").
KNOWN_TYPES = {
    "shape",
    "shape2D",
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
}


def detect_separator(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(path):
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def classify_column(column_name: str) -> str:
    """Returns the 'type' of a feature column: shape, firstorder, glcm,
    glrlm, glszm, gldm, ngtdm or diagnostics. If the pattern isn't
    recognized, returns 'other'."""
    if column_name.startswith("diagnostics"):
        return "diagnostics"

    parts = column_name.split("_")
    if len(parts) >= 2 and parts[1] in KNOWN_TYPES:
        return parts[1]

    # In case a feature had more than one "_" before the type (unusual in
    # pyradiomics, but just in case, check every segment, not only the
    # second one).
    for part in parts:
        if part in KNOWN_TYPES:
            return part

    return "other"


def main():
    input_path = input("Path to the input radiomics CSV: ").strip().strip('"')
    df = load_csv(input_path)

    print(f"\n{input_path}: {len(df)} rows, {len(df.columns)} columns")

    id_column = input(
        "Name of the identifier column to repeat on every sheet "
        "(empty = 'Patient'): "
    ).strip() or "Patient"

    if id_column not in df.columns:
        print(f"WARNING: column '{id_column}' not found in the CSV; "
              "sheets will be generated without an identifier column.")
        id_column = None

    input_path_obj = Path(input_path)
    default_out = str(
        input_path_obj.parent / f"clean_visualization_{input_path_obj.stem}.xlsx"
    )
    output_path = input(
        f"Path to the output .xlsx file (empty = '{default_out}'): "
    ).strip().strip('"') or default_out

    feature_columns = [c for c in df.columns if c != id_column]

    groups = {}
    for col in feature_columns:
        ftype = classify_column(col)
        groups.setdefault(ftype, []).append(col)

    # Sheet order: known types first (in a fixed, readable order), then
    # diagnostics, and finally "other" if anything unrecognized showed up.
    sheet_order = [t for t in ["shape", "shape2D", "firstorder", "glcm", "glrlm",
                                "glszm", "gldm", "ngtdm"] if t in groups]
    if "diagnostics" in groups:
        sheet_order.append("diagnostics")
    if "other" in groups:
        sheet_order.append("other")

    print(f"\nFeature types found ({len(sheet_order)} sheets):")
    for ftype in sheet_order:
        print(f"  - {ftype}: {len(groups[ftype])} columns")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet
        summary = pd.DataFrame(
            [(ftype, len(groups[ftype])) for ftype in sheet_order],
            columns=["type", "num_columns"],
        )
        summary.to_excel(writer, sheet_name="summary", index=False)

        for ftype in sheet_order:
            cols = groups[ftype]
            if id_column:
                sheet_df = pd.concat([df[[id_column]], df[cols]], axis=1)
            else:
                sheet_df = df[cols]
            # Excel limits sheet names to 31 characters
            sheet_name = ftype[:31]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nFile generated: {output_path}")
    print(f"Sheets: summary, {', '.join(sheet_order)}")


if __name__ == "__main__":
    main()
