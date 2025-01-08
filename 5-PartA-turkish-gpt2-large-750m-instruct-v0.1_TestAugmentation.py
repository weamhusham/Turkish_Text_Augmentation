import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from tqdm import tqdm
import torch
import os
import time

# Paths
model_path = r"D:\Important\WeamCollectiveLearning\Project\LLMs\ytu-ce-cosmos-turkish-gpt2-large-750m-instruct-v0.1"
datasets = {
    "news": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\news_test.csv",
    "sentiment": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\sentiment_test.csv"
}
output_dir = r"D:\Important\WeamCollectiveLearning\Project\AugmentedDatasets"

# Percentage of the dataset to process
percentage_to_use = 100  # Set this to a value between 1 and 100

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if torch.cuda.is_available() else -1

print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device_id,
    max_new_tokens=75  # Adjust length for concise paraphrasing
)
print("Model and tokenizer loaded.")

# Function to generate paraphrases for a sentence
def generate_paraphrases(sentence, num_paraphrases):
    paraphrases = []
    for _ in range(num_paraphrases):
        instruction_prompt = (
            f"### Kullanıcı:\nCümleyi tamamen farklı kelimelerle ancak aynı anlamla ifade et: {sentence}\n### Asistan:\n"
        )
        result = text_generator(
            instruction_prompt,
            do_sample=True,
            temperature=0.9,  # Encourage variability
            top_p=0.85,  # Focus on high-probability tokens
            repetition_penalty=1.2  # Avoid repetitive outputs
        )
        generated_response = result[0]['generated_text']
        paraphrase = generated_response[len(instruction_prompt):].strip()
        paraphrases.append(paraphrase)
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
        paraphrases_3 = generate_paraphrases(original_text, num_paraphrases=3)

        # Add records for the 3-augmentations dataset
        for paraphrase in [original_text] + paraphrases_3:
            data_augmented_3.append({text_column: paraphrase, label_column: label})

    print(f"Generating augmentations for {len(subset_data)} records - 5 Paraphrases...")
    for idx, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"{dataset_name} Augmentation - 5 Paraphrases"):
        original_text = row[text_column]
        label = row[label_column]

        # Generate paraphrases for 5 augmentations
        paraphrases_5 = generate_paraphrases(original_text, num_paraphrases=5)

        # Add records for the 5-augmentations dataset
        for paraphrase in [original_text] + paraphrases_5:
            data_augmented_5.append({text_column: paraphrase, label_column: label})

    # Convert to DataFrames
    df_augmented_3 = pd.DataFrame(data_augmented_3)
    df_augmented_5 = pd.DataFrame(data_augmented_5)

    # Save to CSV files
    output_file_3 = os.path.join(output_dir, f"{dataset_name}_GPT2-750m-augmented_3_records.csv")
    output_file_5 = os.path.join(output_dir, f"{dataset_name}_GPT2-750m-augmented_5_records.csv")
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
