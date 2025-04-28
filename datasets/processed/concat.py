import os
import pandas as pd

# 将工作目录切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Find all CSV files with prefix 'train_'
csv_files = ["train_111801.csv", "train_468101.csv", "train_2304601.csv"]

# Read and concatenate all files
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"Added {file} with {len(df)} rows")

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to a new CSV file
output_file = "combined_train.csv"
combined_df.to_csv(output_file, index=False)

print(f"\nConcatenation complete!")
print(f"Total rows in combined dataset: {len(combined_df)}")
print(f"Output saved to: {output_file}")
