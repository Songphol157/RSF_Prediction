# Predict 12-month survival probability with the trained Random Survival Forest
# Python 3.12
# 10/28/2026
# By Steve Tungjitviboonkun, MD MS
#
# Folder structure:
# RSF_Prediction/
# ├── Script/
# │   └── predict_12mo_survival.py
# ├── Data/
# │   ├── data_clinical_template.csv
# │   ├── rsf_model.joblib
# │   └── mode_imputer.joblib        (optional - only needed if predictors have missing values)
# └── Output/
#     └── data_clinical_template_with_12mo_survival.csv
#
# Run:  python Script/predict_12mo_survival.py

import os
import re
import numpy as np
import pandas as pd
import joblib

# --- Paths ---
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "data_clinical_template.csv")
MODEL_PATH = os.path.join(DATA_DIR, "rsf_model.joblib")
IMPUTER_PATH = os.path.join(DATA_DIR, "mode_imputer.joblib")  # optional
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "data_clinical_template_with_12mo_survival.csv")

TIME_POINT = 12  # months
NEW_COL = "12-month survival probability"

for p in (INPUT_CSV, MODEL_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

# --- Same predictor set / encoding as main_script.py ---
predictors = [
    'age_at_diagnosis', 'sex', 'eln2017mode', 'denovo_cat', 'chromosome_cat',
    'flt3_itd_cat', 'npm1_cat', 'runx1_cat', 'asxl1_cat', 'tp53_cat'
]
cat_cols = [
    'sex', 'chromosome_cat', 'denovo_cat',
    'flt3_itd_cat', 'npm1_cat', 'runx1_cat',
    'asxl1_cat', 'tp53_cat'
]

# Template column names -> names used when the model was trained
RENAME_MAP = {
    'age_at_dx': 'age_at_diagnosis',
    'eln_class': 'eln2017mode',
    'is_denovo': 'denovo_cat',
    'karyotype': 'karyotypeconcatenate',
}


def classify_karyo(val):
    """Identical rule to main_script.py: only 46,XX / 46,XY = normal, everything else = complex."""
    if pd.isna(val):
        return np.nan
    if all(re.fullmatch(r'46,(xx|xy)(\[\d+\])?', c)
           for c in re.split(r'[/ ]', str(val).lower().strip())):
        return 'normal'
    return 'complex'


def build_features(raw, feature_names):
    """Clean the raw template and return the model's design matrix (same columns, same order)."""
    df = raw.rename(columns=RENAME_MAP).copy()

    # Lower-case / strip text columns
    for col in [c for c in predictors if c not in ('age_at_diagnosis', 'chromosome_cat')]:
        df[col] = (df[col].astype(str).str.lower().str.strip()
                   .replace({'nan': np.nan, '': np.nan}))

    # De novo: template uses yes/no, training data used true/false
    df['denovo_cat'] = df['denovo_cat'].replace({'yes': 'true', 'no': 'false'})

    # ELN 2017 cleaning + numeric mapping (same as training)
    df['eln2017mode'] = df['eln2017mode'].replace({
        'favorableorintermediate': 'intermediate',
        'intermediateoradverse': 'adverse',
        'missingkaryo': np.nan, 'missingmutations': np.nan, 'noninitial': np.nan,
    })
    df['eln2017mode'] = df['eln2017mode'].map({'favorable': 0, 'intermediate': 1, 'adverse': 2})

    # Karyotype -> chromosome_cat
    df['chromosome_cat'] = df['karyotypeconcatenate'].apply(classify_karyo)

    df['age_at_diagnosis'] = pd.to_numeric(df['age_at_diagnosis'], errors='coerce')

    X = df[predictors].copy()

    # Missing predictors: use the training mode imputer if it is in Data/
    missing_mask = X.isna().any(axis=1)
    if missing_mask.any():
        if os.path.exists(IMPUTER_PATH):
            imputer = joblib.load(IMPUTER_PATH)
            X = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
            print(f"Imputed missing predictors for {missing_mask.sum()} row(s) using mode_imputer.joblib")
            missing_mask[:] = False
        else:
            print(f"WARNING: {missing_mask.sum()} row(s) have missing predictors and no "
                  f"mode_imputer.joblib in Data/ -> their probability will be left blank.")

    # One-hot encode, then force the exact training columns / order
    X = pd.get_dummies(X, columns=cat_cols)
    X = X.reindex(columns=feature_names, fill_value=0)
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    return X, missing_mask


def main():
    raw = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(raw)} rows from {INPUT_CSV}")

    rsf = joblib.load(MODEL_PATH)
    feature_names = list(rsf.feature_names_in_)

    X, missing_mask = build_features(raw, feature_names)

    # Predict row by row: S(12) from each patient's survival curve
    probs = []
    for idx, row in X.iterrows():
        if missing_mask.loc[idx]:
            probs.append(np.nan)
            continue
        surv_fn = rsf.predict_survival_function(row.to_frame().T)[0]
        probs.append(float(surv_fn(TIME_POINT)))

    out = raw.copy()
    out[NEW_COL] = np.round(probs, 4)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(out[['PATIENT_ID', NEW_COL]].head(10).to_string(index=False))
    print(f"\nMean {NEW_COL}: {np.nanmean(probs):.3f}")


if __name__ == "__main__":
    main()
