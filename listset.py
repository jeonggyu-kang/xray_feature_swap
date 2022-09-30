import os
import pandas as pd
import random

source_file = "./cac_sample.csv"
dest_dir = "./data"
train_ratio = 70

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# csv first row
TITLE_ROW = ['dcm_path', 'age', 'sex', 'cac', 'dcm_path2', 'age2', 'sex2', 'cac2']

df = pd.read_csv(source_file)

df = df.loc[:, TITLE_ROW ]

df.dropna(inplace = True)

training = []

for i in range (len(df)):
    num = random.randint(1,1000001)
    training.append(num)
   
    
df['training'] = training

df.sort_values(by=['training'], axis=0)

row_cut = int(len(df)*train_ratio/100)

# df.reset_index()

df = df.loc[:, TITLE_ROW]

df_training = df.iloc[:row_cut,:]
df_test     = df.iloc[row_cut:,:]


df_training.to_parquet(dest_dir+'/train_dataset.parquet')
df_test.to_parquet(dest_dir+'/test_dataset.parquet')