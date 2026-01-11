#!/usr/bin/env python3
"""
explode_codes_from_list.py

Reads a CSV placed in data/Program Data that contains a single column of product codes
(e.g., "CH24/AB/CLR" or "PL100XL/NI") and produces an expanded table with parsed attributes
(Size, Colour, Finish, Region, Glass, Type, IP44, Shade), a best matching new code, Core Code,
Core Code 4, Core Code 4 Family, and the aggregated "Core Code Item Description".

Output CSV is written to the figures directory next to the existing scripts.

This script shares logic with the existing processing but is built to accept a simple
one-column input of codes.

Usage examples
--------------
python explode_codes_from_list.py --input "Codes To Explode.csv"
python explode_codes_from_list.py --input codes.csv --column Code

Notes
-----
- The script autodetects the single column if not specified.
- Expected reference files (in data/Program Data):
    * Old to New Product Codes.csv
    * Unique Product Code Values.csv
    * New Code Unique Product Code Values.csv
    * New Product Code Families.csv (optional; used for Core Code 4 Family)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
from datetime import datetime
import pandas as pd

# ---------------------------
# Paths
# ---------------------------
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path().resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "Program Data"
FIGURE_DIR = PROJECT_ROOT / "figures"

# ---------------------------
# Helpers reused from the pipeline
# ---------------------------

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")


def load_mappings_file(path: Path) -> pd.DataFrame:
    """Load the mapping table and canonicalise headers to ['Original','Mapped','Category'] if present."""
    if not path.exists():
        # fallback: first similarly-named CSV in the folder
        matches = list(path.parent.glob(f"*{path.stem}*.csv"))
        if not matches:
            raise FileNotFoundError(f"Could not find mappings file: {path}")
        path = matches[0]
        print(f"[WARN] '{path.name}' selected as fallback for mappings.")

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # strip whitespace/BOM from headers
    df.rename(columns=lambda c: c.strip().lstrip("\ufeff"), inplace=True)

    # Canonical names
    lower = {c.lower(): c for c in df.columns}
    orig = next((lower[k] for k in ["original", "original product code", "raw", "before"] if k in lower), None)
    mapped = next((lower[k] for k in ["mapped", "stored as", "stored_as", "standard", "cleaned"] if k in lower), None)
    if orig is None:
        raise KeyError(f"Could not find an 'Original' column in {path.name}. Columns: {list(df.columns)}")
    if mapped is None:
        # fallback: last non-Original column
        candidates = [c for c in df.columns if c != orig]
        mapped = next((c for c in candidates if c.lower() == "stored as"), candidates[-1] if candidates else None)
        if mapped is None:
            raise KeyError(f"Could not resolve mapped column in {path.name}.")

    df.rename(columns={orig: "Original", mapped: "Mapped"}, inplace=True)
    # Ensure Category column exists (some files include it, some do not)
    if "Category" not in df.columns:
        df["Category"] = None
    return df


def assign_size(before_slash: str):
    """Return (size_label, remaining_before_slash) or (None, original)."""
    if not isinstance(before_slash, str):
        return None, before_slash
    suffix_size_mapping = {
        "XS": "Extra Small",
        "XL": "Extra Large",
        "S": "Small",
        "M": "Medium",
        "L": "Large",
        "LE": "Large",
        "GT": "Giant",
    }
    for suf, lab in sorted(suffix_size_mapping.items(), key=lambda x: len(x[0]), reverse=True):
        if before_slash.endswith(suf):
            return lab, before_slash[: -len(suf)].strip()
    return None, before_slash


CATEGORY_MAPPING = {
    "Colour": "Colour",
    "Size": "Size",
    "Finish": "Finish",
    "Region": "Region",
    "Glass": "Glass",
    "Type": "Type",
    "IP": "IP44",
    "Shade": "Shade",
}

PREFIX_TO_CATEGORY = {
    "PL": "Pendant",
    "CH": "Chandelier",
    "LA": "Hanging Lantern",
    "WL": "Wall Light",
    "PC": "Pendant Cluster",
    "CL": "Ceiling Light",
    "TL": "Table Lamp",
    "FL": "Floor Lamp",
    "WN": "Wall Lantern",
    "CUSTOM": "Custom",
}


def build_mapping_data(old_new_df: pd.DataFrame, mappings_df: pd.DataFrame) -> pd.DataFrame:
    """Process the Old→New table into a lookup with parsed attributes for scoring."""
    processed = pd.DataFrame()
    processed["Product Code"] = old_new_df["Old Product Code"]
    processed["New Product Code"] = old_new_df["New Product Code"]

    split_res = old_new_df["Old Product Code"].astype(str).str.split("/", n=1, expand=True)
    processed["Code_Before_Slash"] = split_res[0]
    processed["Code_After_Slash"] = split_res[1].fillna("")
    processed["Product Code Prefix"] = processed["Code_Before_Slash"].str[:2]
    processed["Product Category"] = processed["Product Code Prefix"].map(PREFIX_TO_CATEGORY)
    processed["Code_Before_Slash_Original"] = processed["Code_Before_Slash"]

    size_and_rem = processed["Code_Before_Slash"].apply(assign_size)
    processed["Size"] = size_and_rem.apply(lambda x: x[0])
    processed["Code_Before_Slash"] = size_and_rem.apply(lambda x: x[1])

    for col in ["Colour", "Finish", "Region", "Glass", "Type", "IP44", "Shade"]:
        processed[col] = None

    def map_specifiers(code_after: str) -> dict:
        if not code_after:
            return {}
        found = {}
        for tok in str(code_after).split("/"):
            t = tok.strip()
            if not t:
                continue
            rows = mappings_df[mappings_df["Original"].astype(str).str.lower() == t.lower()]
            for _, r in rows.iterrows():
                col = CATEGORY_MAPPING.get(r.get("Category"))
                val = r.get("Mapped", r.get("Stored As"))
                if col and pd.notna(val) and str(val).strip():
                    found.setdefault(col, []).append(str(val))
        return found

    def map_suffixes(code_before: str) -> dict:
        if not code_before:
            return {}
        m = re.search(r"[A-Za-z]+$", str(code_before))
        if not m:
            return {}
        suf = m.group()
        rows = mappings_df[mappings_df["Original"].astype(str).str.strip().str.lower() == suf.lower()]
        found = {}
        for _, r in rows.iterrows():
            col = CATEGORY_MAPPING.get(r.get("Category"))
            val = r.get("Mapped", r.get("Stored As"))
            if col and pd.notna(val) and str(val).strip():
                found.setdefault(col, []).append(str(val))
        return found

    after_maps = processed["Code_After_Slash"].apply(map_specifiers)
    suffix_maps = processed["Code_Before_Slash"].apply(map_suffixes)

    for idx, (a_map, s_map) in enumerate(zip(after_maps, suffix_maps)):
        combo = {**a_map, **s_map}
        for col, vals in combo.items():
            existing = processed.at[idx, col]
            out = []
            if existing:
                out.extend(str(existing).split(", "))
            out.extend(vals)
            # dedupe preserve order
            seen = {}
            processed.at[idx, col] = ", ".join([seen.setdefault(v, v) for v in out if v not in seen])

    return processed


def parse_input_codes(df_codes: pd.DataFrame, mappings_df: pd.DataFrame) -> pd.DataFrame:
    """Parse a single-column input of codes into the same shape expected by the scoring step."""
    # Detect the column
    candidate_cols = [
        "Code", "Product Code", "Old Product Code", "New Product Code",
        "SalesOrderItem.ProductAccountReference", df_codes.columns[0]
    ]
    col = next((c for c in candidate_cols if c in df_codes.columns), df_codes.columns[0])
    series = df_codes[col].astype(str).str.strip()

    out = pd.DataFrame()
    out["Original Product Name"] = series

    split_res = series.str.split("/", n=1, expand=True)
    out["Code_Before_Slash"] = split_res[0].fillna("")
    out["Code_After_Slash"] = split_res[1].fillna("")
    out["Product Code"] = out["Code_Before_Slash"]
    out["Product Code Prefix"] = out["Code_Before_Slash"].str[:2]
    out["Product Category"] = out["Product Code Prefix"].map(PREFIX_TO_CATEGORY)
    out["Code_Before_Slash_Original"] = out["Code_Before_Slash"]

    size_and_rem = out["Code_Before_Slash"].apply(assign_size)
    out["Size"] = size_and_rem.apply(lambda x: x[0])
    out["Code_Before_Slash"] = size_and_rem.apply(lambda x: x[1])

    for colname in ["Colour", "Finish", "Region", "Glass", "Type", "IP44", "Shade"]:
        out[colname] = None

    # Use mappings to populate attributes from tokens after the slash and any trailing alpha suffix
    def map_specifiers(code_after: str) -> dict:
        if not code_after:
            return {}
        found = {}
        for tok in str(code_after).split("/"):
            t = tok.strip()
            if not t:
                continue
            rows = mappings_df[mappings_df["Original"].astype(str).str.lower() == t.lower()]
            for _, r in rows.iterrows():
                col = CATEGORY_MAPPING.get(r.get("Category"))
                val = r.get("Mapped", r.get("Stored As"))
                if col and pd.notna(val) and str(val).strip():
                    found.setdefault(col, []).append(str(val))
        return found

    def map_suffixes(code_before: str) -> dict:
        if not code_before:
            return {}
        m = re.search(r"[A-Za-z]+$", str(code_before))
        if not m:
            return {}
        suf = m.group()
        rows = mappings_df[mappings_df["Original"].astype(str).str.strip().str.lower() == suf.lower()]
        found = {}
        for _, r in rows.iterrows():
            col = CATEGORY_MAPPING.get(r.get("Category"))
            val = r.get("Mapped", r.get("Stored As"))
            if col and pd.notna(val) and str(val).strip():
                found.setdefault(col, []).append(str(val))
        return found

    after_maps = out["Code_After_Slash"].apply(map_specifiers)
    suffix_maps = out["Code_Before_Slash"].apply(map_suffixes)

    for idx, (a_map, s_map) in enumerate(zip(after_maps, suffix_maps)):
        combo = {**a_map, **s_map}
        for colname, vals in combo.items():
            existing = out.at[idx, colname]
            merged = []
            if existing:
                merged.extend(str(existing).split(", "))
            merged.extend(vals)
            # de-duplicate
            seen = {}
            out.at[idx, colname] = ", ".join([seen.setdefault(v, v) for v in merged if v not in seen])

    return out


def match_and_score(df: pd.DataFrame, mapping_data: pd.DataFrame) -> pd.DataFrame:
    """Score against mapping_data to choose a Best Matching New Code (same rules as pipeline)."""
    df = df.copy()
    df["Code Match?"] = False
    df["Size Match?"] = False
    df["Type Match?"] = False
    df["Score"] = 0
    df["Best Matching New Code"] = None
    df["Mapping Data Original Before Product Code"] = None

    for idx, row in df.iterrows():
        cbs = row.get("Code_Before_Slash")
        matched = mapping_data[mapping_data["Code_Before_Slash"] == cbs]
        if matched.empty:
            continue
        matched = matched.copy()
        matched["Score"] = 10
        size_mask = (matched["Size"] == row.get("Size")) | (matched["Size"].isna() & pd.isna(row.get("Size")))
        matched["Score"] += size_mask.astype(int) * 3
        matched["Score"] += (matched["Type"] == row.get("Type")).astype(int) * 1

        df.at[idx, "Code Match?"] = True
        df.at[idx, "Size Match?"] = bool(size_mask.any())
        df.at[idx, "Type Match?"] = bool((matched["Type"] == row.get("Type")).any())

        matched = matched.sort_values("Score", ascending=False)
        if pd.isna(row.get("Size")):
            matched["Code_Length"] = matched["Code_Before_Slash_Original"].astype(str).str.len()
            best = matched.sort_values("Code_Length").iloc[0]
        else:
            best = matched.iloc[0]

        df.at[idx, "Best Matching New Code"] = best.get("New Product Code")
        df.at[idx, "Mapping Data Original Before Product Code"] = best.get("Code_Before_Slash_Original")
        df.at[idx, "Score"] = best["Score"]

    return df


OVERWRITE_COLUMNS = ["Size", "Colour", "Type"]

def parse_new_product_code(new_code: str | None, df_row: pd.Series, mappings_df: pd.DataFrame) -> pd.Series:
    if not isinstance(new_code, str) or not new_code.strip():
        return df_row
    parts = [p.strip() for p in new_code.split("/") if p.strip()]
    if not parts:
        return df_row

    size_candidate = parts[1] if len(parts) > 1 else None
    other_specifiers = parts[2:] if len(parts) > 2 else []

    # size
    if size_candidate:
        rows = mappings_df[mappings_df["Original"].astype(str).str.strip().str.lower() == size_candidate.lower()]
        for _, r in rows.iterrows():
            col = CATEGORY_MAPPING.get(r.get("Category"))
            val = r.get("Mapped", r.get("Stored As"))
            if not col or pd.isna(val) or not str(val).strip():
                continue
            if col in OVERWRITE_COLUMNS:
                df_row[col] = val
            else:
                existing = df_row.get(col, None)
                out = []
                if pd.notna(existing) and isinstance(existing, str):
                    out.extend(v.strip() for v in existing.split(",") if v.strip())
                out.append(str(val))
                # de-dup
                seen = {}
                df_row[col] = ", ".join([seen.setdefault(v, v) for v in out if v not in seen])

    # other specifiers
    for spec in other_specifiers:
        rows = mappings_df[mappings_df["Original"].astype(str).str.strip().str.lower() == spec.lower()]
        for _, r in rows.iterrows():
            col = CATEGORY_MAPPING.get(r.get("Category"))
            val = r.get("Mapped", r.get("Stored As"))
            if not col or pd.isna(val) or not str(val).strip():
                continue
            if col in OVERWRITE_COLUMNS:
                df_row[col] = val
            else:
                existing = df_row.get(col, None)
                out = []
                if pd.notna(existing) and isinstance(existing, str):
                    out.extend(v.strip() for v in existing.split(",") if v.strip())
                out.append(str(val))
                seen = {}
                df_row[col] = ", ".join([seen.setdefault(v, v) for v in out if v not in seen])

    return df_row


def update_df_with_new_codes(df: pd.DataFrame, mappings_df: pd.DataFrame, code_col: str = "Best Matching New Code") -> pd.DataFrame:
    df = df.copy()
    for idx in df.index:
        row = df.loc[idx].copy()
        new_code = row.get(code_col)
        df.loc[idx] = parse_new_product_code(new_code, row, mappings_df)

    df["Core Code"] = df[code_col].astype(str).str.split("/").str[0]
    df["New Product Code Prefix"] = df["Core Code"].astype(str).str[:2]
    df["Product Category"] = df["New Product Code Prefix"].map(PREFIX_TO_CATEGORY)

    drop_cols = [
        "Product Code", "Code_Before_Slash", "Code_After_Slash",
        "Product Code Prefix", "Code_Before_Slash_Original", "anomaly", "Code Match?",
        "Size Match?", "Type Match?", "Mapping Data Original Before Product Code", "New Product Code Prefix",
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def add_product_families_for_code4(df: pd.DataFrame, families_path: Path) -> pd.DataFrame:
    df = df.copy()
    try:
        fam = load_csv(families_path)
    except FileNotFoundError:
        print(f"[WARN] Families file not found: {families_path.name}. Skipping families join.")
        df["Core Code 4 Family"] = None
        return df
    except pd.errors.EmptyDataError:
        print(f"[WARN] Families file is empty: {families_path.name}. Skipping families join.")
        df["Core Code 4 Family"] = None
        return df

    if "Core Code 4" not in df.columns:
        if "Core Code" in df.columns:
            df["Core Code 4"] = df["Core Code"].astype(str).str[:4]
        else:
            df["Core Code 4"] = None

    fam_by_code4 = fam.groupby("Core Code 4")["Family"].first().reset_index()
    out = df.merge(fam_by_code4, on="Core Code 4", how="left")
    out.rename(columns={"Family": "Core Code 4 Family"}, inplace=True)
    return out


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Explode a list of product codes into attributes and best-matching new codes.")
    parser.add_argument("--input", required=True, help="CSV filename in data/Program Data containing a single column of codes")
    parser.add_argument("--column", default=None, help="Name of the column containing the codes (optional; autodetected if omitted)")
    parser.add_argument("--output", default=None, help="Optional explicit output filename (CSV) written to figures/")
    args = parser.parse_args()

    input_path = DATA_DIR / args.input
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # Load reference data
    old_to_new_path = DATA_DIR / "Old to New Product Codes.csv"
    unique_vals_path = DATA_DIR / "Unique Product Code Values.csv"
    new_code_unique_path = DATA_DIR / "New Code Unique Product Code Values.csv"
    families_path = DATA_DIR / "New Product Code Families.csv"

    old_new_df = load_csv(old_to_new_path)
    mappings_df = load_mappings_file(unique_vals_path)
    new_code_unique_df = load_mappings_file(new_code_unique_path)

    # Build mapping table once from Old→New
    mapping_data = build_mapping_data(old_new_df, mappings_df)

    # Load codes
    codes_df = load_csv(input_path)
    if args.column and args.column not in codes_df.columns:
        sys.exit(f"Specified column '{args.column}' not found in {args.input}. Available: {list(codes_df.columns)}")

    if args.column:
        codes_df = codes_df[[args.column]].copy()

    # Parse codes into fields similar to the sales pipeline
    parsed_codes = parse_input_codes(codes_df, mappings_df)

    # Score and pick best matching new code
    scored = match_and_score(parsed_codes, mapping_data)

    # Use the best new code to fill/overwrite attributes as per mapping table for new codes
    final_df = update_df_with_new_codes(scored, new_code_unique_df, code_col="Best Matching New Code")

    # Core Code 4 + Family
    if "Core Code" in final_df.columns:
        final_df["Core Code 4"] = final_df["Core Code"].astype(str).str[:4]
    final_df = add_product_families_for_code4(final_df, families_path)

    # Attach aggregated item descriptions from the Old→New table
    if "Best Matching New Code" in final_df.columns and "New Product Code" in old_new_df.columns and "Item Description" in old_new_df.columns:
        agg_desc = (
            old_new_df.dropna(subset=["Item Description"]).groupby("New Product Code")["Item Description"]
            .apply(lambda s: "; ".join(sorted(pd.unique(s))))
            .reset_index()
            .rename(columns={"New Product Code": "Best Matching New Code", "Item Description": "Core Code Item Description"})
        )
        final_df = final_df.merge(agg_desc, on="Best Matching New Code", how="left")

    # Housekeeping: drop any legacy columns we don't want to expose
    for col in ["New Product Code"]:
        if col in final_df.columns:
            final_df.drop(columns=[col], inplace=True)

    # Save output
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.today().strftime("%Y-%m-%d")
    default_name = f"{input_path.stem} - exploded ({stamp}).csv"
    out_name = args.output if args.output else default_name
    out_path = FIGURE_DIR / out_name
    final_df.to_csv(out_path, index=False)

    print(f"Saved exploded codes to: {out_path}")


if __name__ == "__main__":
    main()
