"""
Data preparation for Federated XAI system.

Dataset: CDC Diabetes Health Indicators (BRFSS 2015)
Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

HOW TO GET THE DATA:
1. Go to https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. Download 'diabetes_binary_health_indicators_BRFSS2015.csv'
3. Place it in this /data folder as 'raw_data.csv'

Then run: python3 prepare_data.py

This script will:
- Load and clean the data
- Create a multi-class target (0=Healthy, 1=Prediabetes/Diabetes, 2=High CVD risk, 3=Hypertension)
- Split into 3 non-overlapping hospital partitions (simulating data silos)
- Save each partition to data/hospital_{A,B,C}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Project root: matches clients/fl_client.py, run_simulation.py (outputs live under data/)
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw_data.csv"

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

TARGET_COL = "disease_class"

CLASS_NAMES = {
    0: "Healthy",
    1: "Diabetes / Prediabetes",
    2: "High Cardiovascular Risk",
    3: "Hypertension Risk"
}


def create_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """
    Derive a 4-class target from the available binary columns.
    Priority order: CVD > Diabetes > Hypertension > Healthy
    CVD takes highest priority so HeartDiseaseorAttack is a clean signal for class 2.
    """
    target = pd.Series(0, index=df.index, name=TARGET_COL)

    # Class 3: Hypertension risk (HighBP + HighChol)
    hypertension_mask = (df["HighBP"] == 1) & (df["HighChol"] == 1)
    target[hypertension_mask] = 3

    # Class 1: Diabetes / prediabetes (overrides hypertension)
    diabetes_col = "Diabetes_binary" if "Diabetes_binary" in df.columns else "Diabetes_012"
    if diabetes_col in df.columns:
        diabetes_mask = df[diabetes_col] > 0
        target[diabetes_mask] = 1

    # Class 2: High CVD risk (HeartDiseaseorAttack or Stroke) — highest priority
    # A patient who has had a heart attack or stroke is CVD class regardless of diabetes status.
    cvd_mask = (df["HeartDiseaseorAttack"] == 1) | (df["Stroke"] == 1)
    target[cvd_mask] = 2

    return target


def split_into_hospital_partitions(df: pd.DataFrame, n_hospitals: int = 3, seed: int = 42):
    """
    Simulate data silos by splitting the dataset into n non-overlapping partitions.
    Each 'hospital' gets a stratified slice of the data.
    """
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    partitions = {}
    partition_size = len(df) // n_hospitals

    hospital_names = ["A", "B", "C"]
    for i, name in enumerate(hospital_names[:n_hospitals]):
        start = i * partition_size
        end = start + partition_size if i < n_hospitals - 1 else len(df)
        partitions[name] = df.iloc[start:end].copy().reset_index(drop=True)

    return partitions


def prepare_and_save():
    if not RAW_FILE.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  RAW DATA NOT FOUND                                          ║
║                                                              ║
║  Please download the dataset:                                ║
║  1. Go to: https://www.kaggle.com/datasets/alexteboul/       ║
║            diabetes-health-indicators-dataset                ║
║  2. Download diabetes_binary_health_indicators_BRFSS2015.csv ║
║  3. Save it as: data/raw_data.csv                            ║
║                                                              ║
║  Then re-run: python3 prepare_data.py                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return False

    print("Loading raw data...")
    df = pd.read_csv(RAW_FILE)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Create multi-class target
    print("Creating multi-class disease target...")
    df[TARGET_COL] = create_multiclass_target(df)

    class_dist = df[TARGET_COL].value_counts().sort_index()
    for cls_id, count in class_dist.items():
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {count:,} samples ({count/len(df)*100:.1f}%)")

    # Keep only the feature columns we need
    available_features = [f for f in FEATURES if f in df.columns]
    df_clean = df[available_features + [TARGET_COL]].dropna()
    print(f"\nClean dataset: {len(df_clean):,} rows, {len(available_features)} features")

    # Fit and save scaler on full data (in real FL this would be done per hospital)
    scaler = StandardScaler()
    scaler.fit(df_clean[available_features])
    scaler_path = DATA_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler → {scaler_path}")

    # Save feature list
    feature_path = DATA_DIR / "features.txt"
    with open(feature_path, "w") as f:
        f.write("\n".join(available_features))
    print(f"Saved feature list → {feature_path}")

    # Split into hospital partitions
    print("\nSplitting into hospital partitions...")
    partitions = split_into_hospital_partitions(df_clean)

    for name, partition in partitions.items():
        out_path = DATA_DIR / f"hospital_{name}.csv"
        partition.to_csv(out_path, index=False)
        print(f"  Hospital {name}: {len(partition):,} rows → {out_path}")

    print("\n✓ Data preparation complete! You can now run the federated learning simulation.")
    return True


if __name__ == "__main__":
    prepare_and_save()
