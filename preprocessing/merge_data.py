import pandas as pd

DROP_COLUMNS = [
    "EcogPtMem",
    "EcogPtLang",
    "EcogPtVisspat",
    "EcogPtPlan",
    "EcogPtOrgan",
    "EcogPtDivatt",
    "EcogPtTotal",
    "EcogSPMem",
    "EcogSPLang",
    "EcogSPVisspat",
    "EcogSPPlan",
    "EcogSPOrgan",
    "EcogSPDivatt",
    "EcogSPTotal",
    "FLDSTRENG",
    "FSVERSION",
    "IMAGEUID",
    "FDG",
    "PIB",
    "AV45",
    "FBB",
    "PTETHCAT",
    "ABETA40",
    "COLPROT",
    "ORIGPROT",
    "PTID",
    "SITE",
    "VISCODE",
    "subject_date",
    "DX_bl",
    "AGE",
    "TAU_adni",
    "PTAU_adni",
    "CDRSB_data",
    "DX",
    "mPACCdigit",
    "mPACCtrailsB",
    "EXAMDATE_bl",
    "CDRSB_bl",
    "ADAS11_bl",
    "ADAS13_bl",
    "ADASQ4_bl",
    "MMSE_bl",
    "RAVLT_immediate_bl",
    "RAVLT_learning_bl",
    "RAVLT_forgetting_bl",
    "RAVLT_perc_forgetting_bl",
    "LDELTOTAL_BL",
    "DIGITSCOR_bl",
    "TRABSCOR_bl",
    "FAQ_bl",
    "mPACCdigit_bl",
    "mPACCtrailsB_bl",
    "FLDSTRENG_bl",
    "FSVERSION_bl",
    "IMAGEUID_bl",
    "Ventricles_bl",
    "Hippocampus_bl",
    "WholeBrain_bl",
    "Entorhinal_bl",
    "Fusiform_bl",
    "MidTemp_bl",
    "ICV_bl",
    "MOCA_bl",
    "EcogPtMem_bl",
    "EcogPtLang_bl",
    "EcogPtVisspat_bl",
    "EcogPtPlan_bl",
    "EcogPtOrgan_bl",
    "EcogPtDivatt_bl",
    "EcogPtTotal_bl",
    "EcogSPMem_bl",
    "EcogSPLang_bl",
    "EcogSPVisspat_bl",
    "EcogSPPlan_bl",
    "EcogSPOrgan_bl",
    "EcogSPDivatt_bl",
    "EcogSPTotal_bl",
    "ABETA_bl",
    "TAU_bl",
    "PTAU_bl",
    "FDG_bl",
    "PIB_bl",
    "AV45_bl",
    "FBB_bl",
    "Years_bl",
    "Month_bl",
    "Month",
    "update_stamp",
    "MMSCORE",
    "FAQ",
    "ABETA42",
    "RID",
    "VSHEIGHT",
    "VSHTUNIT",
    "VSWEIGHT",
    "VSWTUNIT",
    "VSTEMP",
    "VSTMPUNT",
    "VSTMPSRC",
    "ADAS11",
    "ADASQ4",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
]

# Load datasets
meds = pd.read_csv("../ADNI_data.csv")
adnimerge = pd.read_csv("../ADNIMERGE.csv", low_memory=False)

# Optional: Convert date columns (not used for merge, but potentially useful later)
# For EXAMDATE in adnimerge:
# Attempt to parse with a common format, fallback to coercion.
try:
    adnimerge["EXAMDATE"] = pd.to_datetime(adnimerge["EXAMDATE"], format='%Y-%m-%d', errors='raise')
except ValueError:
    adnimerge["EXAMDATE"] = pd.to_datetime(adnimerge["EXAMDATE"], errors='coerce')

# For subject_date in meds:
# Coerce errors as format might be less consistent or unknown.
meds["subject_date"] = pd.to_datetime(meds["subject_date"], errors='coerce')


# Prepare merge keys: convert to string to ensure type compatibility and exact matching.
# This is important for identifiers that might be represented as numbers but should be treated as strings.
if 'subject_id' in meds.columns:
    meds['subject_id'] = meds['subject_id'].astype(str)
else:
    print("Warning: 'subject_id' column not found in ADNI_data.csv (meds). Merge will likely fail or be incorrect.")

if 'visit' in meds.columns:
    meds['visit'] = meds['visit'].astype(str)
else:
    print("Warning: 'visit' column not found in ADNI_data.csv (meds). Merge will likely fail or be incorrect.")

if 'PTID' in adnimerge.columns:
    adnimerge['PTID'] = adnimerge['PTID'].astype(str)
else:
    print("Warning: 'PTID' column not found in ADNIMERGE.csv (adnimerge). Merge will likely fail or be incorrect.")

if 'VISCODE' in adnimerge.columns:
    adnimerge['VISCODE'] = adnimerge['VISCODE'].astype(str)
else:
    print("Warning: 'VISCODE' column not found in ADNIMERGE.csv (adnimerge). Merge will likely fail or be incorrect.")

# Perform the merge
# Using a left merge to keep all records from 'meds' (ADNI_data.csv)
# and add matching information from 'adnimerge' (ADNIMERGE.csv).
# Suffixes are added to distinguish overlapping column names from the two files, if any,
# other than the merge keys.
merged_df = pd.merge(
    meds,
    adnimerge,
    left_on=['subject_id', 'visit'],
    right_on=['PTID', 'VISCODE'],
    how='left',
    suffixes=('_data', '_adni') # Example: if both have 'AGE', results in 'AGE_data', 'AGE_adni'
)

# After merging, PTID and VISCODE from the right table (adnimerge) are somewhat redundant
# if the merge was successful, as subject_id and visit from the left table (meds) are the primary keys.
# You might choose to drop them or inspect them.
# For example, to drop them:
# merged_df = merged_df.drop(columns=['PTID', 'VISCODE'])

# Rename 'M' column to 'months_since_bl'
if 'M' in merged_df.columns:
    merged_df.rename(columns={'M': 'months_since_bl'}, inplace=True)
else:
    print("Warning: Column 'M' not found in merged_df, so it could not be renamed to 'months_since_bl'.")

# Drop specified columns before saving
merged_df = merged_df.drop(columns=DROP_COLUMNS, errors='ignore')

# Save the result
merged_df.to_csv("ADNI_merged.csv", index=False)

print(f"Merged file saved as 'merged_ADNI.csv'")
print(f"Shape of the original meds (ADNI_data.csv) dataframe: {meds.shape}")
print(f"Shape of the original adnimerge (ADNIMERGE.csv) dataframe: {adnimerge.shape}")
print(f"Shape of the merged dataframe: {merged_df.shape}")
print("\nFirst 5 rows of the merged dataframe:")
print(merged_df.head())

# Check for how many rows from meds found a match in adnimerge
# This can be inferred by checking non-null values in a column known to be only in adnimerge (e.g., one of its original keys like PTID if not dropped, or any other adnimerge specific column)
# If 'PTID' was not dropped and was brought into merged_df (as PTID_adni or similar if not a key or due to suffixes)
# or pick a column that is definitely from adnimerge and not in meds
# For example, if 'AGE' is a column in adnimerge and not meds, after merge it would be 'AGE_adni' (or 'AGE' if no conflict)
# Let's assume 'EXAMDATE_adni' (if EXAMDATE was in adnimerge and not meds, or after suffixing) can indicate a match.
# This requires knowing a unique column from adnimerge.
# A simple proxy: count rows where the right keys (now part of merged_df) are not null.
# Since PTID and VISCODE are merge keys from the right, they will be populated if a match is found.
# If they were not dropped, you could use merged_df['PTID'].notna().sum() or merged_df['VISCODE'].notna().sum()
# However, pd.merge with left_on/right_on doesn't automatically rename right_on columns if they conflict with left_on.
# It's better to check a column that is *only* in the right dataframe.
# Let's find a column in adnimerge that is not in meds (excluding the keys we just made strings)
original_adnimerge_cols = set(pd.read_csv("../ADNIMERGE.csv", nrows=0, low_memory=False).columns)
original_meds_cols = set(pd.read_csv("../ADNI_data.csv", nrows=0).columns)
unique_to_adnimerge = list(original_adnimerge_cols - original_meds_cols)

if unique_to_adnimerge:
    # Pick a column that might have been suffixed, e.g. the first unique one
    # The actual name in merged_df would be col_name or col_name + suffix_adni
    example_adnimerge_col = unique_to_adnimerge[0]
    col_in_merged_df_suffixed = example_adnimerge_col + '_adni'
    col_in_merged_df_original = example_adnimerge_col

    actual_col_to_check = None
    if col_in_merged_df_suffixed in merged_df.columns:
        actual_col_to_check = col_in_merged_df_suffixed
    elif col_in_merged_df_original in merged_df.columns:
         actual_col_to_check = col_in_merged_df_original
    
    if actual_col_to_check:
        matches_found = merged_df[actual_col_to_check].notna().sum()
        print(f"Number of rows in ADNI_data.csv that found a match in ADNIMERGE.csv (based on '{actual_col_to_check}'): {matches_found}")
    else:
        print(f"Could not identify a unique column from ADNIMERGE to check for matches (tried {example_adnimerge_col}).")

else:
    print("Could not find a unique column in ADNIMERGE.csv to robustly check merge matches count.")