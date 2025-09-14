import pandas as pd                      #library: pandas
import re                                #remove
from langdetect import detect            #language detection
from tqdm import tqdm                    #display progress bar (loop)
import nltk
from nltk.corpus import stopwords        #remove stop/common words
from nltk.stem import WordNetLemmatizer  #lemmatization/stemming
import emoji                             #detects & remove / replace emoji from text
import glob                              #find file path match to 'commentc....csv

#progress bar
tqdm.pandas()

stop_words = set(stopwords.words('english'))     #load eng stop words
lemmatizer = WordNetLemmatizer()

# 1.handle missing values
def clean_missing(df):
    df = df.dropna(subset=['textOriginal'])
    df['likeCount'] = df['likeCount'].fillna(0).astype(int)  #fill 0 with mising like count
    return df

# 2.remove duplicates
def clean_duplicates(df):
    return df.drop_duplicates(subset=['commentId', 'textOriginal'])

# 3.standardize timestamps
def clean_timestamps(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['updatedAt'] = pd.to_datetime(df['updatedAt'], errors='coerce')
    df['year'] = df['publishedAt'].dt.year     #year >> trend analysis
    df['month'] = df['publishedAt'].dt.month   #month >> seoson analysis
    return df

# 4.language detection
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

def filter_language(df, lang="en"):
    df['lang'] = df['textOriginal'].progress_apply(detect_lang)
    return df[df['lang'] == lang]

# 5.normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   #remove URLs
    text = re.sub(r"@\w+", "", text)      #remove mentions/tags
    text = re.sub(r"[^a-z\s]", "", text)  #keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(df):
    df['clean_text'] = df['textOriginal'].progress_apply(normalize_text)
    return df

# 6.filter spam (basic)
def filter_spam(df):     #not detetct spam words, but filter (remo char <3 (ok, hi, ??))
    return df[df['clean_text'].str.len() > 3]   #calc text length >> keep comments having >3 characters


# 7.remove stopwords (common words)
def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# 8.lemmatization/stemming (reduce to root words)
def lemmatize_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 9.remove emoji (replace with words)
def clean_emojis(text):
    return emoji.replace_emoji(text, replace="")

def advanced_clean(df):
    df['clean_text'] = df['clean_text'].progress_apply(remove_stopwords)
    df['clean_text'] = df['clean_text'].progress_apply(lemmatize_text)
    df['clean_text'] = df['clean_text'].progress_apply(clean_emojis)
    return df

#cleaning pipeline >> sequence of steps (x to call each f(x) manually)
def clean_pipeline(df):
    df = clean_missing(df)
    df = clean_duplicates(df)
    df = clean_timestamps(df)
    df = filter_language(df, lang="en")
    df = clean_text(df)
    df = filter_spam(df)
    df = advanced_clean(df)
    return df

#main script
if __name__ == "__main__":
    #access to the datasets' folder instead of individual files (flexible, scalable)
    folder_path = r"C:\Users\azali\Downloads\Loreal Datathon\datasets" 

    #access to all comment files (comments1.csv ... comments5.csv)
    file_list = glob.glob(folder_path + r"\comments*.csv")   #glob >> find all files names with 'comments*'

    all_chunks = []

    for file_path in file_list:
        print(f"Processing {file_path} ...")
        #process in chunks (100k rows)
        for chunk in pd.read_csv(file_path, chunksize=100000):   #load 100k rows at once
            chunk_clean = clean_pipeline(chunk)   #each chunks goes thru clean pipeline
            all_chunks.append(chunk_clean)        #then append to all cleaned chunks

    #combine all cleaned chunks into one df
    cleaned_df = pd.concat(all_chunks, ignore_index=True)

    #save as parquet: columnar storage file format for big data instead of CSV (smaller & faster than CSV)
    cleaned_df.to_parquet(folder_path + r"\comments_cleaned.parquet", index=False)

    print("Cleaning completed!!!! Saved as comments_cleaned.parquet")
