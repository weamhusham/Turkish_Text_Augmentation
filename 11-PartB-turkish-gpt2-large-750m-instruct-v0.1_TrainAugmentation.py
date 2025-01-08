import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from tqdm import tqdm
import torch
import os
import time

# Paths
model_path = r"D:\Important\WeamCollectiveLearning\Project\LLMs\ytu-ce-cosmos-turkish-gpt2-large-750m-instruct-v0.1"
datasets = {
    "news": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\news_train.csv",
    "sentiment": r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets\sentiment_train.csv"
}
output_dir = r"D:\Important\WeamCollectiveLearning\Project\AugmentedTrainDatasets"

# Percentage of the dataset to process
percentage_to_use = 100  # Use 100% of the dataset

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

    # Generate datasets with 2, 3, and 5 augmentations
    augmentation_configs = [2, 3, 5]
    for num_augmentations in augmentation_configs:
        augmented_data = []
        print(f"Generating {num_augmentations} augmentations for {len(subset_data)} records...")

        for idx, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"{dataset_name} Augmentation - {num_augmentations} Paraphrases"):
            original_text = row[text_column]
            label = row[label_column]

            # Generate paraphrases
            paraphrases = generate_paraphrases(original_text, num_paraphrases=num_augmentations)

            # Add original and augmented records
            for paraphrase in [original_text] + paraphrases:
                augmented_data.append({text_column: paraphrase, label_column: label})

        # Convert to DataFrame
        df_augmented = pd.DataFrame(augmented_data)

        # Save to CSV file
        output_file = os.path.join(output_dir, f"{dataset_name}_GPT2-750m-augmented_{num_augmentations}_records.csv")
        try:
            df_augmented.to_csv(output_file, index=False, encoding="utf-8")
            print(f"Dataset saved: {output_file}")
        except Exception as e:
            print(f"Error saving dataset for '{dataset_name}': {e}")

    # Display timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed processing for {dataset_name} in {elapsed_time:.2f} seconds.")
