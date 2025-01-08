import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import os
import time

# Paths
model_path = r"D:\Important\WeamCollectiveLearning\Project\LLMs\vngrs-ai-VBART-Large-Paraphrasing"
datasets = {
    "news": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\news_test.csv",
    "sentiment": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\sentiment_test.csv"
}
output_dir = r"D:\Important\WeamCollectiveLearning\Project\AugmentedDatasets"

# Percentage of the dataset to process
percentage_to_use = 100  # Set this to a value between 1 and 100

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer from local path, prioritize GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Function to generate multiple paraphrases for a sentence
def generate_paraphrases(sentence, model, tokenizer, num_paraphrases):
    token_input = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Remove token_type_ids if present
    token_input.pop("token_type_ids", None)

    paraphrases = []
    for _ in range(num_paraphrases):
        outputs = model.generate(
            **token_input,
            max_length=150,
            repetition_penalty=3.0,
            length_penalty=1.5,
            num_beams=10,
            do_sample=True,  # Enable sampling
            top_p=0.8,  # Adjust sampling parameters
            temperature=1.3,
            diversity_penalty=0.0,  # Disable diversity penalty
            num_beam_groups=1  # Single beam group
        )

        paraphrases.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return paraphrases

# Process each dataset
for dataset_name, dataset_path in datasets.items():
    print(f"Processing dataset: {dataset_name}")
    start_time = time.time()

    # Load dataset
    try:
        data = pd.read_csv(dataset_path, delimiter=",")
        if dataset_name == "news":
            text_column = "HABERLER"
            label_column = "ETIKET"
        elif dataset_name == "sentiment":
            text_column = "review"
            label_column = "score"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if text_column not in data.columns or label_column not in data.columns:
            raise ValueError(f"Expected columns '{text_column}' and '{label_column}' not found in the dataset.")
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        continue
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        continue

    # Use a subset of the dataset based on the percentage
    subset_data = data.sample(frac=percentage_to_use / 100, random_state=42).reset_index(drop=True)

    # Generate datasets with 3 and 5 augmentations
    data_augmented_3 = []
    data_augmented_5 = []

    print(f"Generating augmentations for {len(subset_data)} records...")
    for idx, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"{dataset_name} Augmentation - 3 Paraphrases"):
        original_text = row[text_column]
        label = row[label_column]

        # Generate paraphrases for 3 augmentations
        paraphrases_3 = generate_paraphrases(original_text, model, tokenizer, num_paraphrases=3)

        # Add records for the 3-augmentations dataset
        for paraphrase in [original_text] + paraphrases_3:
            data_augmented_3.append({text_column: paraphrase, label_column: label})

    print(f"Generating augmentations for {len(subset_data)} records - 5 Paraphrases...")
    for idx, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"{dataset_name} Augmentation - 5 Paraphrases"):
        original_text = row[text_column]
        label = row[label_column]

        # Generate paraphrases for 5 augmentations
        paraphrases_5 = generate_paraphrases(original_text, model, tokenizer, num_paraphrases=5)

        # Add records for the 5-augmentations dataset
        for paraphrase in [original_text] + paraphrases_5:
            data_augmented_5.append({text_column: paraphrase, label_column: label})

    # Convert to DataFrames
    df_augmented_3 = pd.DataFrame(data_augmented_3)
    df_augmented_5 = pd.DataFrame(data_augmented_5)

    # Save to CSV files
    output_file_3 = os.path.join(output_dir, f"{dataset_name}_VBART-Large-augmented_3_records.csv")
    output_file_5 = os.path.join(output_dir, f"{dataset_name}_VBART-Large-augmented_5_records.csv")
    try:
        df_augmented_3.to_csv(output_file_3, index=False, encoding="utf-8")
        df_augmented_5.to_csv(output_file_5, index=False, encoding="utf-8")
        print(f"Datasets saved for {dataset_name}:")
        print(f"  3 Augmentations: {output_file_3}")
        print(f"  5 Augmentations: {output_file_5}")
    except Exception as e:
        print(f"Error saving datasets for '{dataset_name}': {e}")

    # Display timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed processing for {dataset_name} in {elapsed_time:.2f} seconds.")
