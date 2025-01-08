import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# Paths
processed_path = r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets"
augmented_path = r"D:\Important\WeamCollectiveLearning\Project\AugmentedTrainDatasets"
output_path = r"D:\Important\WeamCollectiveLearning\Project\Embeddings"
model_path = r"D:\Important\WeamCollectiveLearning\Project\Models\ytu-ce-cosmos-turkish-colbert"
results_path = r"D:\Important\WeamCollectiveLearning\Project\ResultsTrain"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Logging utility
def log_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Adjust batch size based on GPU memory
def adjust_batch_size():
    free_memory = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    if free_memory > 6 * 1024 ** 3:
        return 64
    elif free_memory > 4 * 1024 ** 3:
        return 32
    elif free_memory > 2 * 1024 ** 3:
        return 16
    else:
        return 8

# Preprocess text specifically for ColBERT
def preprocess_colbert_text(texts):
    return [
        str(text).replace("I", "ı").lower() if isinstance(text, str) else ""
        for text in texts
    ]

# Generate embeddings
def generate_embeddings(texts, transformer_model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model.to(device)
    transformer_model.eval()

    texts = preprocess_colbert_text(texts)
    embeddings = []
    batch_size = adjust_batch_size()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = [str(text) for text in texts[i:i + batch_size] if text.strip()]
            if not batch_texts:
                continue

            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = transformer_model(**inputs)

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output.to(torch.float32).cpu().numpy()
            else:
                batch_embeddings = outputs.last_hidden_state[:, 0, :].to(torch.float32).cpu().numpy()

            embeddings.append(batch_embeddings)

    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0,), dtype=np.float32)

# Train and evaluate ML model
def train_and_evaluate_model(train_embeddings, train_labels, test_embeddings, test_labels, model_type):
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "adaboost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50),
        "svm": SVC(kernel="linear", probability=True, max_iter=1000),
        "xgboost": xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", use_label_encoder=False),
        "lightgbm": lgb.LGBMClassifier(verbose=-1),
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = models[model_type]
    log_timestamp(f"Training {model_type} model...")
    model.fit(train_embeddings, train_labels)
    log_timestamp(f"Training completed for {model_type}.")

    log_timestamp("Evaluating model...")
    predictions = model.predict(test_embeddings)
    metrics = {
        "Accuracy": accuracy_score(test_labels, predictions),
        "Precision": precision_score(test_labels, predictions, average="weighted"),
        "Recall": recall_score(test_labels, predictions, average="weighted"),
        "F1-Score": f1_score(test_labels, predictions, average="weighted"),
    }
    log_timestamp(f"Evaluation Metrics: {metrics}")
    return metrics

# Process datasets
def process_datasets(datasets_group, tokenizer, transformer_model, test_dataset):
    test_file_path = str(os.path.join(test_dataset[3], test_dataset[0]))
    log_timestamp(f"Loading test dataset: {test_file_path}")

    assert os.path.isfile(test_file_path), f"Test file not found: {test_file_path}"

    test_data = pd.read_csv(test_file_path, encoding="utf-8")
    test_texts = test_data[test_dataset[1]].tolist()
    test_labels = test_data[test_dataset[2]].tolist()
    test_embeddings = generate_embeddings(test_texts, transformer_model, tokenizer)

    results = []
    for dataset in datasets_group:
        train_file_path = str(os.path.join(dataset[3], dataset[0]))
        log_timestamp(f"Processing training dataset: {train_file_path}")

        assert os.path.isfile(train_file_path), f"Training file not found: {train_file_path}"

        train_data = pd.read_csv(train_file_path, encoding="utf-8")
        train_texts = train_data[dataset[1]].tolist()
        train_labels = train_data[dataset[2]].tolist()
        train_embeddings = generate_embeddings(train_texts, transformer_model, tokenizer)

        for model_type in ["random_forest", "adaboost", "svm", "xgboost", "lightgbm"]:
            metrics = train_and_evaluate_model(train_embeddings, train_labels, test_embeddings, test_labels, model_type)
            results.append({
                "Dataset": dataset[0],
                "Model": model_type,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
            })

    return results

# Save results to CSV
def save_results_table(results, results_path):
    log_timestamp("Saving results to CSV...")
    output_file = os.path.join(results_path, "training_augmentation_results_ytu-ce-cosmos-colbert.csv")
    results_df = pd.DataFrame(results)

    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_file, index=False)

    log_timestamp(f"Results saved to {output_file}")

# Main execution
def main():
    datasets = [
        ("news_train.csv", "HABERLER", "ETIKET", processed_path),
        ("news_test.csv", "HABERLER", "ETIKET", processed_path),
        ("sentiment_train.csv", "review", "score", processed_path),
        ("sentiment_test.csv", "review", "score", processed_path),
        ("news_GPT2-750m-augmented_2_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_GPT2-750m-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_GPT2-750m-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-Large-augmented_2_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-Large-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-Large-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-XLarge-augmented_2_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-XLarge-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-XLarge-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("sentiment_GPT2-750m-augmented_2_records.csv", "review", "score", augmented_path),
        ("sentiment_GPT2-750m-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_GPT2-750m-augmented_5_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-Large-augmented_2_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-Large-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-Large-augmented_5_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-XLarge-augmented_2_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-XLarge-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-XLarge-augmented_5_records.csv", "review", "score", augmented_path),
    ]

    test_datasets = [
        ("news_test.csv", "HABERLER", "ETIKET", processed_path),
        ("sentiment_test.csv", "review", "score", processed_path),
    ]

    log_timestamp("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    for test_dataset in test_datasets:
        train_datasets = [d for d in datasets if d[1] == test_dataset[1] and d[2] == test_dataset[2] and d != test_dataset]
        results = process_datasets(train_datasets, tokenizer, model, test_dataset)
        save_results_table(results, results_path)

if __name__ == "__main__":
    main()
