import pandas as pd
from pathlib import Path
from pathlib import Path

v2 = [2, 4, 5, 10,12,14,15,16, 18, 19,20]  # Example V2 values for the plot



for id in v2:
    df = pd.read_csv(f"data/dataset5/{id}.txt", header=None, delimiter='\t')
    ecg = df.iloc[:,0-1].values.astype(float)
    pd.Series(ecg).to_csv(f"data/dataset5/{id}.csv", index=False, header=False)