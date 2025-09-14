import pandas as pd

df = pd.read_parquet(r"C:\Users\azali\Downloads\Loreal Datathon\datasets\comments_cleaned.parquet")

print(df.shape)    #total rows & columns
print(df.info())
print(df.head(30))  #sample first 10 rows

