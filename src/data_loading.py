import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def load_raw_matches () -> pd.DataFrame:
    """ Load and combine all CSV match files found in data/raw."""
    
    raw_dir = DATA_DIR / "raw"
    csv_files = list(raw_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in data/raw duirectory.")
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["SeasonFile"] = file.name # track source file
        dfs.append(df)
        
    combined = pd.concat(dfs, ignore_index=True)
    return combined