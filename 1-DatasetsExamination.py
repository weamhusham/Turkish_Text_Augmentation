import pandas as pd
from collections import Counter

# Paths to datasets
sentiment_path = r"D:\Important\WeamCollectiveLearning\Project\Maydogan-TRSAv1"
news_path = r"D:\Important\WeamCollectiveLearning\Project\turkish-news_dataset"


# Function to examine CSV files
def examine_csv(file_path, dataset_name, label_column=None):
    print(f"\n{'-' * 40}")
    print(f"Examining {dataset_name} dataset: {file_path}")
    print(f"{'-' * 40}")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("\nSample Data (First 5 Rows):")
    print(df.head())
    print("\nMissing Values (Per Column):")
    print(df.isnull().sum())

    # Examine label distribution if label_column is provided
    if label_column and label_column in df.columns:
        print(f"\nLabel Distribution ({label_column}):")
        label_counts = Counter(df[label_column])
        for label, count in label_counts.items():
            print(f"  {label}: {count} instances")
    return df


# Examine Turkish Sentiment Analysis Dataset
print("\nTurkish Sentiment Analysis Dataset V1 Examination")
sentiment_train = examine_csv(f"{sentiment_path}/TRSAv1.csv", "Turkish Sentiment", label_column="score")

# Examine Turkish News Dataset
print("\nTurkish News Dataset Examination")
news_dataset = examine_csv(f"{news_path}/TurkishHeadlines.csv", "Turkish News Headlines", label_column="ETIKET")
