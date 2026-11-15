"""
Data acquisition module for PULSE.
Downloads the de-identified medical appointment no-shows dataset
and verifies integrity.
"""
import os
import urllib.request
import pandas as pd
import hashlib

DATA_URL = "https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/master/noshowappointments-kagglev2-may-2016.csv"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "noshowappointments-kagglev2-may-2016.csv")

def download_dataset(url: str = DATA_URL, destination: str = RAW_DATA_PATH, force: bool = False) -> str:
    """
    Downloads raw dataset from URL to local destination if not already present.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination) and not force:
        print(f"[INFO] Raw dataset already exists at: {destination}")
    else:
        print(f"[INFO] Downloading dataset from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (PULSE Project Data Ingestion)'})
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[SUCCESS] Download completed ({len(data)} bytes written).")

    # Verification
    df = pd.read_csv(destination)
    print("\n--- Dataset Ingestion Verification ---")
    print(f"Path: {destination}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().to_dict()}")
    return destination

if __name__ == "__main__":
    download_dataset()
