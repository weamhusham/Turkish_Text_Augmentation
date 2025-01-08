import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

# Paths to datasets
sentiment_path = r"D:\Important\WeamCollectiveLearning\Project\Maydogan-TRSAv1"
news_path = r"D:\Important\WeamCollectiveLearning\Project\turkish-news_dataset"
output_path = r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation, lowercase, and strip whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Lowercase and trim spaces
    return text

# Function to filter out rows containing any emoji
def filter_rows_with_emojis(df, text_column):
    print(f"\nFiltering rows containing emojis in column '{text_column}'...")
    df[text_column] = df[text_column].fillna('').str.strip()

    # Define a regex pattern to match any emoji
    emoji_pattern = re.compile(
        r"["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE
    )

    # Filter rows where the text contains any emoji
    df = df[~df[text_column].str.contains(emoji_pattern, regex=True)]

    print(f"Remaining rows after filtering: {len(df)}")
    return df

# Function to split dataset into train and test sets
def split_dataset(df, label_column, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=42)
    print(f"\nTrain Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    return train_df, test_df

# Function to process the sentiment dataset
def process_sentiment_dataset(file_path, train_output, test_output, target_size):
    print(f"\nProcessing Sentiment Dataset: {file_path}")
    df = pd.read_csv(file_path)

    # Drop the 'id' column if it exists
    if 'id' in df.columns:
        print("Dropping the 'id' column...")
        df = df.drop(columns=['id'])

    df['review'] = df['review'].apply(preprocess_text)  # Preprocess text
    df = filter_rows_with_emojis(df, text_column="review")  # Remove rows containing emojis

    # Downsample sentiment dataset to match target size
    if len(df) > target_size:
        print(f"Downsampling sentiment dataset from {len(df)} to {target_size}...")
        df = df.sample(n=target_size, random_state=42).reset_index(drop=True)

    train_df, test_df = split_dataset(df, label_column="score")  # Split into train and test

    # Save the processed datasets
    train_df.to_csv(f"{output_path}/{train_output}", index=False)
    test_df.to_csv(f"{output_path}/{test_output}", index=False)
    print(f"Saved train dataset to {output_path}/{train_output}")
    print(f"Saved test dataset to {output_path}/{test_output}")

# Function to process the news dataset and return its size
def process_news_dataset(file_path, train_output, test_output):
    print(f"\nProcessing News Dataset: {file_path}")
    df = pd.read_csv(file_path)
    df['HABERLER'] = df['HABERLER'].apply(preprocess_text)  # Preprocess text
    df = filter_rows_with_emojis(df, text_column="HABERLER")  # Remove rows containing emojis

    train_df, test_df = split_dataset(df, label_column="ETIKET")  # Split into train and test

    # Save the processed datasets
    train_df.to_csv(f"{output_path}/{train_output}", index=False)
    test_df.to_csv(f"{output_path}/{test_output}", index=False)
    print(f"Saved train dataset to {output_path}/{train_output}")
    print(f"Saved test dataset to {output_path}/{test_output}")

    # Return the number of rows in the news dataset
    return len(df)

# Process News Dataset to determine its size
news_size = process_news_dataset(f"{news_path}/TurkishHeadlines.csv", "news_train.csv", "news_test.csv")

# Process Sentiment Dataset with target size matching the news dataset
process_sentiment_dataset(f"{sentiment_path}/TRSAv1.csv", "sentiment_train.csv", "sentiment_test.csv", target_size=news_size)
