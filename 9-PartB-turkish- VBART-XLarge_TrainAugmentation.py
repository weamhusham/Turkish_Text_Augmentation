import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import os
import time

# Paths
model_path = r"D:\Important\WeamCollectiveLearning\Project\LLMs\vngrs-ai-VBART-XLarge-Paraphrasing"
datasets = {
    "news": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\news_train.csv",
    "sentiment": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\sentiment_train.csv"
}
output_dir = r"D:\Important\WeamCollectiveLearning\Project\AugmentedTrainDatasets"

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

    # Generate datasets with 2, 3, and 5 augmentations
    augmentation_configs = [2, 3, 5]
    for num_augmentations in augmentation_configs:
        data_augmented = []

        print(f"Generating {num_augmentations} augmentations for {len(subset_data)} records...")
        for idx, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"{dataset_name} Augmentation - {num_augmentations} Paraphrases"):
            original_text = row[text_column]
            label = row[label_column]

            # Generate paraphrases for the current augmentation count
            paraphrases = generate_paraphrases(original_text, model, tokenizer, num_paraphrases=num_augmentations)

            # Add records for the augmented dataset
            for paraphrase in [original_text] + paraphrases:
                data_augmented.append({text_column: paraphrase, label_column: label})

        # Convert to DataFrame and save
        df_augmented = pd.DataFrame(data_augmented)
        output_file = os.path.join(output_dir, f"{dataset_name}_VBART-XLarge-augmented_{num_augmentations}_records.csv")
        try:
            df_augmented.to_csv(output_file, index=False, encoding="utf-8")
            print(f"Dataset saved: {output_file}")
        except Exception as e:
            print(f"Error saving dataset for '{dataset_name}' with {num_augmentations} augmentations: {e}")

    # Display timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed processing for {dataset_name} in {elapsed_time:.2f} seconds.")
