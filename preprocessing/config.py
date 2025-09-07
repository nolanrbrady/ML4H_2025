import os

# =======================================================================================================================
# CONFIGURATION FOR PREPROCESSING
# =======================================================================================================================

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'preprocessing')
MODEL_TRAINING_DIR = os.path.join(ROOT_DIR, 'model_training')
CLINICIAN_POLICY_DIR = os.path.join(ROOT_DIR, 'clinician_policy')
ALPACA_DIR = os.path.join(ROOT_DIR, 'reinforcement_learning', 'ALPACA')

# --- Feature Definitions ---
COLUMNS_TO_DROP = [
    'VSRESP', 'VSPULSE', 'TOTALMOD', 'FAQTOTAL', 'PTEDUCAT', 'MMSE', 'TRABSCOR', 'ADNI_EF',
    'DIGITSCOR', 'LDELTOTAL', 'RAVLT_immediate', 'GDTOTAL', 'MOCA', 'VSBPDIA', 'GLUCOSE', 'PROTEIN', 'CTRED', 'CTWHITE', 
    'VSBPSYS', 'ADAS13', 'CDRSB_adni', 'PTEDUCAT', 'PTAU_data', 'FAQTOTAL', 'TOTALMOD'
]

CONTINUOUS_VARS = [
    'ADNI_MEM', 'ADNI_EF', 'ADNI_EF2',
    'VSBPDIA', 'VSPULSE', 'VSRESP', 'GLUCOSE', 'PROTEIN', 'CTRED', 'CTWHITE',
    'TAU_data', 'PTAU_data', 'VSBPSYS', 'subject_age', 'ABETA', 'TRABSCOR',
    'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
    'MidTemp', 'ICV', 'months_since_bl'
]

ORDINAL_VARS = [
    'GDTOTAL', 'TOTALMOD', 'FAQTOTAL', 'PTEDUCAT', 'CDRSB_adni', 'ADAS13',
    'MMSE', 'RAVLT_immediate', 'LDELTOTAL', 'DIGITSCOR', 'MOCA'
]

CATEGORICAL_VARS_FOR_IMPUTATION = ['PTGENDER', 'PTRACCAT', 'PTMARRY', 'research_group']

DROP_VARS_FOR_IMPUTATION = [
    'CMBGNYR_DRVD', 'CMENDYR_DRVD', 'CMENDYR_DRVD_filled', 'visit', 'CMROUTE', 
    'CMREASON', 'CMUNITS', 'CMMED', 'GENOTYPE', 'EXAMDATE', 'DIAGNOSIS', 
    'DXNORM', 'DXMCI', 'DXDEP', 'CMMED_clean', 'med_class', 'visit_year', 'APOE4'
]

ACTION_FEATURES = [
    "AD Treatment_active", "Alpha Blocker_active", "Analgesic_active", 
    "Antidepressant_active", "Antihypertensive_active", "Bone Health_active", 
    "Diabetes Medication_active", "Diuretic_active", "NSAID_active", 
    "No Medication_active", "Other_active", "PPI_active", "SSRI_active", 
    "Statin_active", "Steroid_active", "Supplement_active", "Thyroid Hormone_active"
]

DRUG_CLASS_MAPPING = {
    'aricept': 'AD Treatment', 'donepezil': 'AD Treatment', 'namenda': 'AD Treatment', 'exelon': 'AD Treatment',
    'lipitor': 'Statin', 'simvastatin': 'Statin', 'crestor': 'Statin', 'zocor': 'Statin', 'atorvastatin': 'Statin',
    'lisinopril': 'Antihypertensive', 'atenolol': 'Antihypertensive', 'amlodipine': 'Antihypertensive',
    'metoprolol': 'Antihypertensive', 'norvasc': 'Antihypertensive', 'losartan': 'Antihypertensive',
    'levothyroxine': 'Thyroid Hormone', 'synthroid': 'Thyroid Hormone',
    'aspirin': 'NSAID', 'ibuprofen': 'NSAID', 'aleve': 'NSAID', 'asa': 'NSAID',
    'tylenol': 'Analgesic', 'acetaminophen': 'Analgesic',
    'zoloft': 'SSRI', 'lexapro': 'SSRI', 'sertraline': 'SSRI', 'citalopram': 'SSRI',
    'trazodone': 'Antidepressant', 'prozac': 'SSRI',
    'metformin': 'Diabetes Medication',
    'vitamin d': 'Supplement', 'vitamin d3': 'Supplement', 'vitamin b12': 'Supplement',
    'vitamin c': 'Supplement', 'vitamin e': 'Supplement', 'calcium': 'Supplement',
    'multivitamin': 'Supplement', 'fish oil': 'Supplement',
    'omeprazole': 'PPI', 'prilosec': 'PPI',
    'hydrochlorothiazide': 'Diuretic',
    'fosamax': 'Bone Health',
    'prednisone': 'Steroid', 'prednisolone': 'Steroid',
    'flomax': 'Alpha Blocker',
    'no medication': 'No Medication'
}