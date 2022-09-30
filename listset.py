import os
import pandas as pd
import random
import pickle

source_file = "./cac_sample.csv"
dest_dir = "./data"
train_ratio = 70

# destination parquet
dest_train_parquet = 'train_dataset.parquet'
dest_test_parquet = 'test_dataset.parquet'
dest_pickle = 'title.pkl'

os.makedirs(dest_dir,exist_ok=True)

df = pd.read_csv(source_file)

# dump title info.
title_list = list(df.columns) # file_name age ...
with open(os.path.join(dest_dir, dest_pickle), 'wb') as f:
    pickle.dump(title_list , f)

df = df.loc[: , title_list]
df.dropna(inplace = True)

# shuffle samples
df = df.sample(frac=1).reset_index(drop=True)

row_cut = int(len(df)*train_ratio/100)
df_training = df.iloc[:row_cut,:]
df_test     = df.iloc[row_cut:,:]


df_training.to_parquet(os.path.join(dest_dir, dest_train_parquet))
df_test.to_parquet(os.path.join(dest_dir, dest_test_parquet))

print ('Parquet & pickle files have written in {}!'.format(dest_dir))