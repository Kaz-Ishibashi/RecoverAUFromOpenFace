import pandas as pd
import sys
import os

def drop_au_columns(input_path, output_path):
    try:
        df = pd.read_csv(input_path, skipinitialspace=True)
        
        # Identify AU columns (start with 'AU' and usually end with _r or _c)
        # OpenFace columns: AU01_r, AU01_c, etc.
        au_cols = [col for col in df.columns if col.strip().startswith('AU')]
        
        if not au_cols:
            print(f"No AU columns found in {input_path}")
        else:
            print(f"Dropping {len(au_cols)} AU columns: {au_cols}")
            df.drop(columns=au_cols, inplace=True)
            
        df.to_csv(output_path, index=False)
        print(f"Saved stripped CSV to: {output_path}")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python drop_au_columns.py <input_csv> <output_csv>")
        sys.exit(1)
        
    drop_au_columns(sys.argv[1], sys.argv[2])
