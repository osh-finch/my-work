import pandas as pd
import re
from pathlib import Path
import numpy as np
from datetime import datetime

def run_transformation(file_path: str, old_new_raw: pd.DataFrame, mappings_df: pd.DataFrame, new_code_unique_df: pd.DataFrame):
    """
    High-level function that:
      1) Loads 'Old to New Product Codes.csv' plus 'Unique Product Code Values.csv'
         to build a processed "mapping_data" DataFrame.
      2) Loads the data from `file_path` (the file you want to transform) and matches/scores
         each row against the above "mapping_data".
      3) Loads 'New Code Unique Product Code Values.csv' to parse and update columns
         according to the 'Best Matching New Code'.
      4) Returns the final transformed DataFrame.

    Parameters
    ----------
    file_path : str
        The absolute path to the CSV file to be processed.

    Returns
    -------
    pd.DataFrame
        The final transformed DataFrame with columns updated or appended.
    """

    # ----------------------------------------------------------------
    # 0. Configuration
    # ----------------------------------------------------------------
    # If running in a script file, we can capture __file__. If in Jupyter, fallback:
    try:
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        script_dir = Path().resolve()
    data_dir = script_dir.parent / 'data' / 'Program Data'

    # ----------------------------------------------------------------
    # 1. Utility: CSV loader
    # ----------------------------------------------------------------
    def load_and_prepare_data_internal(file_path):
        encoding = 'ISO-8859-1'
        return pd.read_csv(file_path, encoding=encoding)

    # ----------------------------------------------------------------
    # 2. Process "Old to New Product Codes" into a mapping_data DataFrame
    # ----------------------------------------------------------------
    def assign_size_internal(before_slash):
        suffix_size_mapping = {
            'XS': 'Extra Small',
            'XL': 'Extra Large',
            'S': 'Small',
            'M': 'Medium',
            'L': 'Large',
            'LE': 'Large',
            'GT': 'Giant'
        }
        # Check longer suffixes first so 'XL' isn't confused as 'L'
        for suffix, size_val in sorted(suffix_size_mapping.items(), key=lambda x: len(x[0]), reverse=True):
            if before_slash.endswith(suffix):
                # Remove the matched suffix
                new_before_slash = before_slash[: -len(suffix)].strip()
                return size_val, new_before_slash
        return None, before_slash

    def process_data_internal(df, mappings_df):
        """
        Processes 'Old Product Code' in df:
          - Splits code before/after slash
          - Deduces size from suffix
          - Maps suffixes & slash-based specifiers from 'Unique Product Code Values.csv'.
        """
        processed_df = pd.DataFrame()
        processed_df['Product Code'] = df['Old Product Code']
        processed_df["New Product Code"] = df["New Product Code"]

        split_res = df['Old Product Code'].str.split('/', n=1, expand=True)
        processed_df['Code_Before_Slash'] = split_res[0]
        processed_df['Code_After_Slash'] = split_res[1].fillna('')

        # Derive product category from first 2 letters
        prefix_to_category = {
            'PL': 'Pendant',
            'CH': 'Chandelier',
            'LA': 'Hanging Lantern',
            'WL': 'Wall Light',
            'PC': 'Pendant Cluster',
            'CL': 'Ceiling Light',
            'TL': 'Table Lamp',
            'FL': 'Floor Lamp',
            'WN': 'Wall Lantern',
            'CUSTOM': 'Custom'
        }
        processed_df['Product Code Prefix'] = processed_df['Code_Before_Slash'].str[:2]
        processed_df['Product Category'] = processed_df['Product Code Prefix'].map(prefix_to_category)
        processed_df['Code_Before_Slash_Original'] = processed_df['Code_Before_Slash']

        # Assign Size
        size_and_remainder = processed_df['Code_Before_Slash'].apply(assign_size_internal)
        processed_df['Size'] = size_and_remainder.apply(lambda x: x[0])
        processed_df['Code_Before_Slash'] = size_and_remainder.apply(lambda x: x[1])

        # Initialize these columns if they don't exist
        for col in ["Colour", "Finish", "Region", "Glass", "Type", "IP44","Shade"]:
            processed_df[col] = None

        # Category mapping
        cat_map = {
            'Colour': 'Colour',
            'Size': 'Size',
            'Finish': 'Finish',
            'Region': 'Region',
            'Glass': 'Glass',
            'Type': 'Type',
            'IP': 'IP44',
            "Shade": "Shade",
        }

        # --- Helper: map slash-based specifiers ---
        def map_specifiers(code_after_slash):
            if not code_after_slash:
                return {}
            specifiers = code_after_slash.split('/')
            found_values = {}
            for spec in specifiers:
                spec_clean = spec.strip()
                if not spec_clean:
                    continue
                match_rows = mappings_df[mappings_df['Original'].str.lower() == spec_clean.lower()]
                if not match_rows.empty:
                    for _, row_map in match_rows.iterrows():
                        category_in_csv = row_map['Category']
                        stored_as = row_map.get('Mapped', row_map.get('Stored As'))
                        col_to_update = cat_map.get(category_in_csv)
                        if col_to_update:
                            found_values.setdefault(col_to_update, []).append(stored_as)
            return found_values

        # --- Helper: map the longest letter-based suffix in Code_Before_Slash ---
        def map_suffixes(code_before_slash):
            if not code_before_slash:
                return {}
            suffix_match = re.search(r'[A-Za-z]+$', code_before_slash)
            if not suffix_match:
                return {}
            longest_suffix = suffix_match.group()
            match_rows = mappings_df[
                mappings_df['Original'].str.strip().str.lower() == longest_suffix.lower()
            ]
            found_values = {}
            if not match_rows.empty:
                for _, row_map in match_rows.iterrows():
                    category_in_csv = row_map['Category']
                    stored_as = row_map.get('Mapped', row_map.get('Stored As'))
                    col_to_update = cat_map.get(category_in_csv)
                    if col_to_update:
                        found_values.setdefault(col_to_update, []).append(stored_as)
            return found_values

        # Apply mapping to after_slash and suffix
        all_after = processed_df['Code_After_Slash'].apply(map_specifiers)
        all_suffix = processed_df['Code_Before_Slash'].apply(map_suffixes)

        for idx, (after_dict, suffix_dict) in enumerate(zip(all_after, all_suffix)):
            combined = {**after_dict, **suffix_dict}
            for colx, val_list in combined.items():
                existing_val = processed_df.at[idx, colx]
                updated = []
                if existing_val:
                    updated.extend(existing_val.split(', '))
                updated.extend(val_list)
                updated = list(dict.fromkeys(updated))  # deduplicate
                processed_df.at[idx, colx] = ', '.join(updated)

        return processed_df

    # ----------------------------------------------------------------
    # 3. Scoring: match_and_score
    # ----------------------------------------------------------------
    def match_and_score(df, mapping_data):
        """
        Compare rows in `df` to `mapping_data` and compute a Score.
        - +10 for exact Code_Before_Slash match
        - +3 if sizes match (including NaN==NaN)
        - +1 if types match
        - If Size in df is NaN, pick the match with the shortest Code_Before_Slash_Original.
        """
        df["Code Match?"] = False
        df["Size Match?"] = False
        df["Type Match?"] = False
        df["Score"] = 0
        df["Best Matching New Code"] = None
        df["Mapping Data Original Before Product Code"] = None

        for index, row in df.iterrows():
            cbs = row.get("Code_Before_Slash", None)
            matched_rows = mapping_data[mapping_data["Code_Before_Slash"] == cbs]
            if not matched_rows.empty:
                matched_rows = matched_rows.copy()
                matched_rows["Score"] = 10
                size_match_mask = (
                    (matched_rows["Size"] == row["Size"]) |
                    (matched_rows["Size"].isna() & pd.isna(row["Size"]))
                )
                matched_rows["Score"] += size_match_mask.astype(int) * 3
                matched_rows["Score"] += (matched_rows["Type"] == row["Type"]).astype(int) * 1

                df.at[index, "Code Match?"] = True
                df.at[index, "Size Match?"] = size_match_mask.any()
                df.at[index, "Type Match?"] = (matched_rows["Type"] == row["Type"]).any()

                matched_rows = matched_rows.sort_values("Score", ascending=False)
                if pd.isna(row["Size"]):
                    matched_rows["Code_Length"] = matched_rows["Code_Before_Slash_Original"].str.len()
                    best_match = matched_rows.sort_values("Code_Length").iloc[0]
                else:
                    best_match = matched_rows.iloc[0]

                df.at[index, "Best Matching New Code"] = best_match.get("New Product Code", None)
                df.at[index, "Mapping Data Original Before Product Code"] = best_match.get("Code_Before_Slash_Original", None)
                df.at[index, "Score"] = best_match["Score"]

        return df

    # ----------------------------------------------------------------
    # 4. parse_new_product_code & update_df_with_new_codes
    # ----------------------------------------------------------------
    OVERWRITE_COLUMNS = ["Size", "Colour", "Type"]
    CATEGORY_MAPPING = {
        'Colour': 'Colour',
        'Size': 'Size',
        'Finish': 'Finish',
        'Region': 'Region',
        'Glass': 'Glass',
        'Type': 'Type',
        'IP': 'IP44',
        "Shade": "Shade"
    }

    def parse_new_product_code(new_code, df_row, mappings_df):
        """Use 'new_code' to update df_row columns from mappings_df data."""
        if not isinstance(new_code, str) or not new_code.strip():
            return df_row
        parts = [p.strip() for p in new_code.split('/') if p.strip()]
        if not parts:
            return df_row

        # part[0] is core, part[1] might be size, part[2:] other specifiers
        size_candidate = parts[1] if len(parts) > 1 else None
        other_specifiers = parts[2:] if len(parts) > 2 else []

        # 1) size_candidate
        if size_candidate:
            match_rows = mappings_df[
                mappings_df['Original'].str.strip().str.lower() == size_candidate.lower()
            ]
            if not match_rows.empty:
                for _, match_row in match_rows.iterrows():
                    cat = match_row['Category']
                    new_val = match_row.get('Mapped', match_row.get('Stored As'))
                    if pd.isna(new_val) or not str(new_val).strip():
                        continue
                    col = CATEGORY_MAPPING.get(cat)
                    if col:
                        if col in OVERWRITE_COLUMNS:
                            df_row[col] = new_val
                        else:
                            existing_val = df_row.get(col, None)
                            out_list = []
                            if pd.notna(existing_val) and isinstance(existing_val, str):
                                out_list.extend(val.strip() for val in existing_val.split(',') if val.strip())
                            out_list.append(str(new_val))
                            out_list = list(dict.fromkeys(out_list))
                            df_row[col] = ', '.join(out_list)

        # 2) other specifiers
        for spec in other_specifiers:
            sclean = spec.strip()
            if not sclean:
                continue
            mrows = mappings_df[mappings_df['Original'].str.strip().str.lower() == sclean.lower()]
            if mrows.empty:
                continue
            for _, match_row in mrows.iterrows():
                cat = match_row['Category']
                new_val = match_row.get('Mapped', match_row.get('Stored As'))
                if pd.isna(new_val) or not str(new_val).strip():
                    continue
                col = CATEGORY_MAPPING.get(cat)
                if col:
                    if col in OVERWRITE_COLUMNS:
                        df_row[col] = new_val
                    else:
                        existing_val = df_row.get(col, None)
                        out_list = []
                        if pd.notna(existing_val) and isinstance(existing_val, str):
                            out_list.extend(val.strip() for val in existing_val.split(',') if val.strip())
                        out_list.append(str(new_val))
                        out_list = list(dict.fromkeys(out_list))
                        df_row[col] = ', '.join(out_list)
        return df_row

    def add_product_families_for_code4(df, families_path):
        """
        Load New Product Code Families.csv and join with DataFrame to add Core Code 4 Family.
        """
        try:
            # Load the families file
            families_df = pd.read_csv(families_path)
            
            # Create Core Code 4 column if it doesn't exist
            if 'Core Code 4' not in df.columns and 'Core Code' in df.columns:
                df['Core Code 4'] = df['Core Code'].str[:4]
            elif 'Core Code 4' not in df.columns:
                 # Cannot proceed if Core Code doesn't exist to create Core Code 4
                 print("Warning: 'Core Code' column not found, cannot create 'Core Code 4 Family'.")
                 df['Core Code 4 Family'] = None
                 return df

            # Join for Core Code 4 (first 4 characters match)
            # Group by Core Code 4 to get one family per Core Code 4
            families_code4 = families_df.groupby('Core Code 4')['Family'].first().reset_index()
            df = df.merge(
                families_code4,
                on='Core Code 4',
                how='left'
            )
            
            # Rename the merged 'Family' column
            df.rename(columns={'Family': 'Core Code 4 Family'}, inplace=True)
            
            return df
                
        except FileNotFoundError:
            print(f"Warning: Families file not found: {families_path}")
            df['Core Code 4 Family'] = None
            return df
        except pd.errors.EmptyDataError:
            print(f"Warning: Families file is empty: {families_path}")
            df['Core Code 4 Family'] = None
            return df
        except Exception as e:
            print(f"Error loading families file {families_path}: {e}")
            df['Core Code 4 Family'] = None
            return df

    def update_df_with_new_codes(df, mappings_df, code_col="Best Matching New Code"):
        """
        Iterates each row in df, parses `code_col` to update columns from 'mappings_df'.
        Then creates a "Core Code" col with everything before the first '/'.
        Drops columns no longer needed.
        """
        # Define the prefix-to-category mapping within the function
        prefix_to_category = {
            'PL': 'Pendant',
            'CH': 'Chandelier',
            'LA': 'Hanging Lantern',
            'WL': 'Wall Light',
            'PC': 'Pendant Cluster',
            'CL': 'Ceiling Light',
            'TL': 'Table Lamp',
            'FL': 'Floor Lamp',
            'WN': 'Wall Lantern',
            'CUSTOM': 'Custom'
        }

        for idx in df.index:
            row_copy = df.loc[idx].copy()
            new_code = row_copy[code_col]
            updated_row = parse_new_product_code(new_code, row_copy, mappings_df)
            df.loc[idx] = updated_row

        # Add "Core Code" = everything before first slash
        df["Core Code"] = df[code_col].str.split('/').str[0]
        # Map Prefix to Category
        df['New Product Code Prefix'] = df['Core Code'].str[:2]
        df['Product Category'] = df['New Product Code Prefix'].map(prefix_to_category)

        # Drop unneeded columns if they exist
        to_drop = [
            "Product Code", "Code_Before_Slash", "Code_After_Slash",
            "Product Code Prefix", "Code_Before_Slash_Original", "anomaly", "Code Match?",
            "Size Match?", "Type Match?", "Mapping Data Original Before Product Code", 'New Product Code Prefix'
        ]
        df = df.drop(columns=to_drop, errors="ignore")
        return df

    # ----------------------------------------------------------------
    # 5. Main Orchestration
    # ----------------------------------------------------------------

    # 5a. Load "Old to New Product Codes" -> processed into "mapping_data"
    old_to_new_path = data_dir / "Old to New Product Codes.csv"
    old_new_raw = load_and_prepare_data_internal(old_to_new_path)
    unique_vals_path = data_dir / "Unique Product Code Values.csv"
    unique_vals_df = load_and_prepare_data_internal(unique_vals_path)

    # Build our main "mapping_data" from the "Old to New Product Codes"
    processed_product_codes = process_data_internal(old_new_raw, mappings_df)

    # 5b. Load the user-specified data from file_path
    df_to_transform = load_and_prepare_data_internal(file_path)

    # 5c. Match & score with processed_product_codes
    scored_df = match_and_score(df_to_transform, processed_product_codes)

    # 5e. Update rows with "Best Matching New Code"
    final_df = update_df_with_new_codes(scored_df, new_code_unique_df, code_col="Best Matching New Code")

    # 5f. Add Product Code 4 families information
    families_path = data_dir / "New Product Code Families.csv"
    final_df = add_product_families_for_code4(final_df, families_path)

    # 5g. Consolidate and add Item Descriptions
    if "Best Matching New Code" in final_df.columns:
        cols = {c.strip(): c for c in old_new_raw.columns}
        if "New Product Code" in cols and "Item Description" in cols:

            # ① Group by code: aggregate unique descriptions, sort them, and join with "; "
            agg_descriptions = (
                old_new_raw
                .dropna(subset=[cols["Item Description"]])
                .groupby(cols["New Product Code"])[cols["Item Description"]]
                .apply(lambda s: "; ".join(sorted(pd.unique(s))))
                .reset_index()
                .rename(columns={
                    cols["New Product Code"]: "Best Matching New Code",
                    cols["Item Description"]: "Core Code Item Description_new"
                })
            )

            # ② Drop existing description columns from the main df to avoid conflicts
            if "Core Code Item Description" in final_df.columns:
                final_df = final_df.drop(columns=["Core Code Item Description"])

            # ③ Merge the new aggregated descriptions
            final_df = final_df.merge(
                agg_descriptions,
                on="Best Matching New Code",
                how="left"
            )
            # ④ Rename the new column to the final name
            final_df.rename(columns={"Core Code Item Description_new": "Core Code Item Description"}, inplace=True)

    # Final cleanup before returning
    cols_to_drop = ['Core Code Family', 'New Product Code']
    final_df.drop(columns=[col for col in cols_to_drop if col in final_df.columns], inplace=True)

    # Return the fully updated DataFrame
    return final_df

def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """Load CSV (with ISO-8859-1 encoding), parse dates and numeric quantities, sort by date."""
    encoding = 'ISO-8859-1'
    df = pd.read_csv(file_path, encoding=encoding)
    df['SalesOrder.Date'] = pd.to_datetime(df['SalesOrder.Date'], format='%d/%m/%Y', errors='coerce')
    df['SalesOrderItem.Quantity'] = pd.to_numeric(df['SalesOrderItem.Quantity'], errors='coerce')
    df.dropna(subset=['SalesOrderItem.Quantity'], inplace=True)
    df.sort_values(by='SalesOrder.Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def assign_size(before_slash: str):
    """
    Derive size from the text before the first slash.
    Checks for a known suffix and returns (size, remaining_text).
    """
    suffix_size_mapping = {
        'XS': 'Extra Small',
        'XL': 'Extra Large',
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large',
        'LE': 'Large',
        'GT': 'Giant'
    }
    # Check longer suffixes first to avoid partial matches (e.g. "XL" vs. "L")
    for suffix, size in sorted(suffix_size_mapping.items(), key=lambda x: len(x[0]), reverse=True):
        if before_slash.endswith(suffix):
            return size, before_slash[:-len(suffix)].strip()
    return None, before_slash

def process_data(df: pd.DataFrame, file_name: str, valid_product_codes_list: list, mappings_df: pd.DataFrame):
    """
    Processes raw DataFrame to extract and structure information from the
    'SalesOrderItem.ProductAccountReference' column.
    """
    # Create initial dataframe with required columns
    processed_df = pd.DataFrame({
        'Order Code': df['SalesOrderItem.ProductAccountReference'] + '/',
        'Date': df['SalesOrder.Date'],
        'Order Number': df['SalesOrder.Number'],
        'Quantity': df['SalesOrderItem.Quantity'],
        'Each?': df['SalesOrderItem.UnitOfSale'],
        'Net Value': df['SalesOrderItem.AmountNet'],
        'Product Code': df['SalesOrderItem.ProductAccountReference'].str.split('/').str[0],
        'Original Product Name': df['SalesOrderItem.ProductAccountReference'], # Renamed and kept full original
    })

    # Clean and convert 'Net Value' to a numeric type, removing any non-numeric characters like apostrophes.
    processed_df['Net Value'] = pd.to_numeric(
        processed_df['Net Value'].astype(str).str.replace("'", "", regex=False).str.replace(",", "", regex=False),
        errors='coerce'
    )
    processed_df.dropna(subset=['Net Value'], inplace=True)

    split_results = processed_df['Original Product Name'].str.split('/', n=1, expand=True)
    processed_df['Code_Before_Slash'] = split_results[0].fillna('')
    processed_df['Code_After_Slash'] = split_results[1].fillna('')

    # Validate product codes
    valid_mask = processed_df['Code_Before_Slash'].apply(
        lambda code: any(valid_code.startswith(code) for valid_code in valid_product_codes_list)
    )
    valid_items = processed_df[valid_mask].copy()
    dropped_items = processed_df[~valid_mask].copy()
    dropped_items.reset_index(drop=True, inplace=True)

    # Map product code prefix to category
    prefix_to_category = {
        'PL': 'Pendant',
        'CH': 'Chandelier',
        'LA': 'Hanging Lantern',
        'WL': 'Wall Light',
        'PC': 'Pendant Cluster',
        'CL': 'Ceiling Light',
        'TL': 'Table Lamp',
        'FL': 'Floor Lamp',
        'CUSTOM': 'Custom'
    }
    valid_items['Product Code Prefix'] = valid_items['Code_Before_Slash'].str[:2]
    valid_items['Product Category'] = valid_items['Product Code Prefix'].map(prefix_to_category)
    valid_items['Code_Before_Slash_Original'] = valid_items['Code_Before_Slash']

    # Assign size and update Code_Before_Slash
    sizes_and_codes = valid_items['Code_Before_Slash'].apply(assign_size)
    valid_items['Size'] = sizes_and_codes.apply(lambda x: x[0])
    valid_items['Code_Before_Slash'] = sizes_and_codes.apply(lambda x: x[1])

    # Initialize additional attribute columns
    for col in ["Colour", "Finish", "Region", "Glass", "Type", "IP44", "Shade"]:
        if col not in valid_items.columns:
            valid_items[col] = None

    # Mapping for specifiers after slash
    category_mapping = {
        'Colour': 'Colour',
        'Size': 'Size',
        'Finish': 'Finish',
        'Region': 'Region',
        'Glass': 'Glass',
        'Type': 'Type',
        'IP': 'IP44',
        'Shade': 'Shade'
    }

    def map_specifiers(code_after_slash: str) -> dict:
        """Map specifiers after slash using the CSV mappings."""
        if not code_after_slash:
            return {}
        found_values = {}
        for spec in code_after_slash.split('/'):
            spec_clean = spec.strip()
            if not spec_clean:
                continue
            match_rows = mappings_df[mappings_df['Original'].str.lower() == spec_clean.lower()]
            for _, row in match_rows.iterrows():
                cat_in_csv = row['Category']
                stored_val = row.get('Mapped', row.get('Stored As'))
                col_to_update = category_mapping.get(cat_in_csv)
                if col_to_update:
                    found_values.setdefault(col_to_update, []).append(stored_val)
        return found_values

    # Map the specifiers and update columns
    all_found_values = valid_items['Code_After_Slash'].apply(map_specifiers)
    for idx, found_dict in all_found_values.items():
        for col_name, vals in found_dict.items():
            existing = valid_items.at[idx, col_name]
            combined = (existing.split(', ') if pd.notnull(existing) else []) + vals
            # Remove duplicates while preserving order
            valid_items.at[idx, col_name] = ', '.join(dict.fromkeys(combined))

    # Adjust net value (depending on unit of sale)
    valid_items['Adjusted Net Value'] = valid_items["Net Value"]
    valid_items.drop(columns=["Each?", "Net Value"], inplace=True)

    return valid_items, dropped_items

def filter_core_code_range(df: pd.DataFrame, prefix: str, start_num: int, end_num: int) -> pd.DataFrame:
    """Filter df where 'Core Code' starts with prefix and its numeric part is within range."""
    def within_range(core_code: str) -> bool:
        if not core_code.startswith(prefix):
            return False
        numeric_part = core_code[len(prefix):]
        return numeric_part.isdigit() and start_num <= int(numeric_part) <= end_num

    return df[df['Core Code'].apply(within_range)].copy()

def filter_by_codes_and_ranges(df: pd.DataFrame, user_entries: list) -> pd.DataFrame:
    """
    Filter DataFrame rows by product codes or ranges.
    user_entries: list of strings like ["CH24", "PL100-PL119", "WL10"].
    """
    pattern = re.compile(r'^([A-Za-z]+)(\d+)-([A-Za-z]+)?(\d+)$')
    matched_frames = []

    for entry in (e.strip() for e in user_entries):
        m = pattern.match(entry)
        if m:
            prefix1, start_str, prefix2, end_str = m.groups()
            prefix = prefix1 if not prefix2 or prefix2 != prefix1 else prefix1
            try:
                start_num, end_num = sorted([int(start_str), int(end_str)])
            except ValueError:
                print(f"Could not parse numeric range in '{entry}'. Skipping.")
                continue
            range_filtered = filter_core_code_range(df, prefix, start_num, end_num)
            matched_frames.append(range_filtered)
            print(f"Applied range filter: '{entry}' => {len(range_filtered)} rows matched.")
        else:
            sub_df = df[df['Core Code'].str.startswith(entry)].copy()
            matched_frames.append(sub_df)
            print(f"Applied prefix filter: '{entry}' => {len(sub_df)} rows matched.")

    if not matched_frames:
        print("No valid filters or matches found. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(matched_frames, ignore_index=True).drop_duplicates()
    print(f"Total rows after combining filters: {len(out)}")
    return out

def remove_upper_outliers_quantile_isolation_forest(
    df: pd.DataFrame, remove_outliers: bool = True, column_name: str = 'Adjusted Net Value', contamination: float = 0.01
):
    """Remove upper-tail outliers using an 'inverted' IsolationForest approach."""
    from sklearn.ensemble import IsolationForest # Imported here to avoid circular dependency if data_processing is imported by main
    if not remove_outliers or df.empty:
        return df.copy(), pd.DataFrame()
    X = df[[column_name]].copy()
    X_inv = X.max() - X
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(X_inv)
    df['anomaly'] = preds
    non_outliers = df[df['anomaly'] == 1].drop(columns='anomaly')
    outliers = df[df['anomaly'] == -1].drop(columns='anomaly')
    print(f"Removing upper-tail outliers (IsolationForest). Contamination={contamination}")
    print(f"Outliers removed: {len(outliers)} | Non-outliers remain: {len(non_outliers)}")
    return non_outliers.copy(), outliers.copy()

def enforce_display_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce display rules for Core Code and Core Code 4:
    - Core Code: Show full code (all digits) for every unique entry
    - Core Code 4: Truncate to first 4 digits, group by prefix
    """
    # Ensure Core Code shows full code for every unique entry
    if 'Core Code' in df.columns:
        # Convert to string and ensure all digits are preserved
        df['Core Code'] = df['Core Code'].astype(str).str.strip()
    
    # Ensure Core Code 4 is properly truncated to 4 digits
    if 'Core Code' in df.columns:
        df['Core Code 4'] = df['Core Code'].str[:4]
    
    return df

def create_grouped_dataframes(df: pd.DataFrame) -> dict:
    """Group data by time (year, month, quarter) and by each attribute column for both value and quantity."""
    if df.empty:
        print("Warning: DataFrame is empty. Returning empty grouped results.")
        return {}

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().all():
        print("Error: 'Date' column contains invalid or missing datetime values.")
        return {}

    df = enforce_display_rules(df)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.to_period('Q')
    df['Month.Year.Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2), errors='coerce')
    df.sort_values(by='Month.Year.Date', inplace=True)

    categories = [
        'Product Category', 'Size', 'Finish', 'Colour', 'Region', 'Glass', 'Type', 'IP44', 'Core Code', 'Core Code 4'
    ]
    
    grouped_dfs = {
        'overall': {
            'year': df.groupby('Year').agg({'Adjusted Net Value': 'sum', 'Quantity': 'sum'}).reset_index(),
            'month': df.groupby('Month.Year.Date').agg({'Adjusted Net Value': 'sum', 'Quantity': 'sum'}).reset_index(),
            'quarter': df.groupby('Quarter').agg({'Adjusted Net Value': 'sum', 'Quantity': 'sum'}).reset_index(),
        }
    }
    
    # This block is now redundant as quantity is included in the main groupings, but we keep it
    # to avoid breaking any downstream code that might specifically look for these keys.
    grouped_dfs['overall_quantity'] = {
        'year': df.groupby('Year')['Quantity'].sum().reset_index(),
        'month': df.groupby('Month.Year.Date')['Quantity'].sum().reset_index(),
        'quarter': df.groupby('Quarter')['Quantity'].sum().reset_index(),
    }

    # Define base aggregations
    agg_value_and_quantity = {'Adjusted Net Value': 'sum', 'Quantity': 'sum'}

    # Custom aggregation function to join unique descriptions
    def aggregate_descriptions(series):
        all_descs = set()
        for item in series.dropna():
            # Split items that might already be joined, and add individual parts to the set
            for part in str(item).split(';'):
                all_descs.add(part.strip())
        # Return a sorted, semi-colon-joined string of unique descriptions
        return '; '.join(sorted(list(all_descs)))

    # Define columns to keep and their aggregation methods
    cols_to_keep = {
        'Core Code 4 Family': 'first',
        'Core Code Item Description': aggregate_descriptions
    }

    # Dynamically create aggregation dictionaries
    final_agg_dict = agg_value_and_quantity.copy()
    for col, agg_func in cols_to_keep.items():
        if col in df.columns:
            final_agg_dict[col] = agg_func

    for cat in categories:
        # --- Value-based groupings (now includes Quantity) ---
        if cat in ['Core Code', 'Core Code 4']:
            grouped_dfs[cat] = {
                'overall': df.groupby(cat, dropna=False).agg(final_agg_dict).reset_index(),
                'year': df.groupby([cat, 'Year'], dropna=False).agg(final_agg_dict).reset_index(),
                'month': df.groupby([cat, 'Month.Year.Date'], dropna=False).agg(final_agg_dict).reset_index(),
                'quarter': df.groupby([cat, 'Quarter'], dropna=False).agg(final_agg_dict).reset_index(),
            }
        else:
            grouped_dfs[cat] = {
                'overall': df.groupby(cat, dropna=False).agg(agg_value_and_quantity).reset_index(),
                'year': df.groupby([cat, 'Year'], dropna=False).agg(agg_value_and_quantity).reset_index(),
                'month': df.groupby([cat, 'Month.Year.Date'], dropna=False).agg(agg_value_and_quantity).reset_index(),
                'quarter': df.groupby([cat, 'Quarter'], dropna=False).agg(agg_value_and_quantity).reset_index(),
            }

        # --- Create quantity-specific views from the combined dataframes ---
        quantity_key = f"{cat}_quantity"
        grouped_dfs[quantity_key] = {}
        for period, data in grouped_dfs[cat].items():
            # MODIFICATION: Instead of selecting a subset of columns, copy the entire dataframe.
            # This ensures that 'Adjusted Net Value' is retained in quantity-focused dataframes,
            # making it available for CSV export.
            grouped_dfs[quantity_key][period] = data.copy()
    
    return grouped_dfs


def project_with_regression_yearly(years, values, future_years=1):
    """Simple linear regression to predict future sales in subsequent years."""
    from sklearn.linear_model import LinearRegression # Imported here to avoid circular dependency
    if len(years) < 2:
        return []
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    last_year = years[-1]
    future_X = np.array([last_year + i for i in range(1, future_years + 1)]).reshape(-1, 1)
    return model.predict(future_X)

def compute_statistics(grouped_dfs: dict) -> dict:
    """Compute monthly, quarterly, yearly growth stats and store them in a dictionary."""
    stats_results = {}
    overall_data = grouped_dfs.get('overall', {})
    overall_stats = {}

    # Monthly statistics
    if 'month' in overall_data and not overall_data['month'].empty:
        monthly_df = overall_data['month'].copy().sort_values('Month.Year.Date')
        monthly_df['Pct_Change'] = monthly_df['Adjusted Net Value'].pct_change() * 100
        monthly_df['Pct_Change'].replace([np.inf, -np.inf], np.nan, inplace=True)
        overall_stats['Average Monthly Growth %'] = monthly_df['Pct_Change'].mean()

    # Quarterly statistics
    if 'quarter' in overall_data and not overall_data['quarter'].empty:
        quarterly_df = overall_data['quarter'].copy()
        quarterly_df['Quarter'] = quarterly_df['Quarter'].astype(str)
        quarterly_df = quarterly_df.sort_values('Quarter')
        quarterly_df['Pct_Change'] = quarterly_df['Adjusted Net Value'].pct_change() * 100
        quarterly_df['Pct_Change'].replace([np.inf, -np.inf], np.nan, inplace=True)
        overall_stats['Average Quarterly Growth %'] = quarterly_df['Pct_Change'].mean()

    # Yearly statistics
    if 'year' in overall_data and not overall_data['year'].empty:
        yearly_df = overall_data['year'].copy().sort_values('Year')
        yearly_df['Pct_Change'] = yearly_df['Adjusted Net Value'].pct_change() * 100
        yearly_df['Pct_Change'].replace([np.inf, -np.inf], np.nan, inplace=True)
        yoy_dict = yearly_df.set_index('Year')['Pct_Change'].to_dict()
        overall_stats['Year-over-Year Growth %'] = yoy_dict
        years = yearly_df['Year'].values
        vals = yearly_df['Adjusted Net Value'].values
        if len(years) >= 2:
            predicted = project_with_regression_yearly(years, vals, future_years=1)
            if predicted.size > 0 and vals[-1] != 0:
                overall_stats[f'Projected Growth {years[-1] + 1} %'] = ((predicted[0] - vals[-1]) / vals[-1]) * 100
            else:
                overall_stats[f'Projected Growth {years[-1] + 1} %'] = None

    stats_results['Overall'] = overall_stats
    return stats_results

def validate_core_code_datasets(grouped_dfs: dict) -> dict:
    """
    Validate Core Code vs Core Code Sales datasets.
    Count unique codes in Sales and Quantity and report any discrepancies.
    
    Returns:
        dict: Validation report with counts and missing codes
    """
    validation_report = {
        'Core Code': {
            'sales_unique_count': 0,
            'quantity_unique_count': 0,
            'missing_in_quantity': [],
            'missing_in_sales': []
        },
        'Core Code 4': {
            'sales_unique_count': 0,
            'quantity_unique_count': 0,
            'missing_in_quantity': [],
            'missing_in_sales': []
        }
    }
    
    for category in ['Core Code', 'Core Code 4']:
        # Get sales data
        sales_key = category
        quantity_key = f"{category}_quantity"
        
        if sales_key in grouped_dfs and 'overall' in grouped_dfs[sales_key]:
            sales_df = grouped_dfs[sales_key]['overall']
            sales_codes = set(sales_df[category].dropna().astype(str))
            validation_report[category]['sales_unique_count'] = len(sales_codes)
        else:
            sales_codes = set()
            
        if quantity_key in grouped_dfs and 'overall' in grouped_dfs[quantity_key]:
            quantity_df = grouped_dfs[quantity_key]['overall']
            quantity_codes = set(quantity_df[category].dropna().astype(str))
            validation_report[category]['quantity_unique_count'] = len(quantity_codes)
        else:
            quantity_codes = set()
        
        # Find missing codes
        validation_report[category]['missing_in_quantity'] = sorted(list(sales_codes - quantity_codes))
        validation_report[category]['missing_in_sales'] = sorted(list(quantity_codes - sales_codes))
    
    # Validation report removed as requested - now runs silently
    return validation_report

def apply_attribute_whitelists(df: pd.DataFrame, whitelists: dict) -> pd.DataFrame:
    """Filter df so each attribute's value is in the whitelisted set (if provided)."""
    if not whitelists:
        print("No advanced attribute whitelists specified.")
        return df
    filtered = df.copy()
    for attr, vals in whitelists.items():
        if vals:
            # Separate NaN from other values in the whitelist
            has_nan_in_whitelist = any(pd.isna(v) for v in vals)
            other_vals = [v for v in vals if pd.notna(v)]

            # Create a boolean mask for the filter
            mask = pd.Series(False, index=filtered.index)
            
            # If there are regular values, check for them
            if other_vals:
                mask = mask | filtered[attr].isin(other_vals)
            
            # If NaN is in the whitelist, check for NaN values in the column
            if has_nan_in_whitelist:
                mask = mask | filtered[attr].isna()
            
            filtered = filtered[mask]
            
            # For printing, represent nan nicely
            vals_str = ['N/A' if pd.isna(v) else str(v) for v in vals]
            print(f"Filtering on {attr}={vals_str}, {len(filtered)} rows remain.")
    return filtered
