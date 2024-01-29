import os
import pandas as pd
import random

root_folder = '/home/ada/satmae/other_data/eurosat/EuroSAT_MS'

folders = sorted(os.listdir(root_folder))
categories = {folder: idx + 1 for idx, folder in enumerate(folders)}
print(folders)
print(categories)

data = []
for folder in folders:
    folder_path = os.path.join(root_folder, folder)
    print(folder_path)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        files = [i for i in files if i.endswith("tif")]
        files = [os.path.join(folder_path, i) for i in files]
        for file in files:
            data.append({'category': categories[folder], 'image_path': file})

print("data:", data)
# Create a DataFrame
df = pd.DataFrame(data)
print(df.head())

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Splitting into train/val
train_size = int(0.35 * len(df))
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]
print(train_df.shape,val_df.shape)
#print(train_df['category'].nunique(), val_df['category'].nunique())
print(df.head())

# Save train and val DataFrames to CSV
train_df.to_csv('train_ms.csv', index=False)
val_df.to_csv('val_ms.csv', index=False)

# Save the entire DataFrame to CSV
df.to_csv('full_dataset_ms.csv', index=False)
