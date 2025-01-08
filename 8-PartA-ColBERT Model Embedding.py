import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

model_name = "ytu-ce-cosmos-ColBERT"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Paths
processed_path = r"D:\Important\WeamCollectiveLearning\Project\ProcessedDatasets"
augmented_path = r"D:\Important\WeamCollectiveLearning\Project\AugmentedTrainDatasets"
output_path = r"D:\Important\WeamCollectiveLearning\Project\Embeddings"
model_path = r"D:\Important\WeamCollectiveLearning\Project\Models\ytu-ce-cosmos-turkish-colbert"
results_path = r"D:\Important\WeamCollectiveLearning\Project\ResultsTest"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Helper function for logging timestamps
def log_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def preprocess_colbert_text(texts):
    """Preprocess text for ColBERT: Handle Turkish-specific issues and lowercase."""
    return [
        str(text).replace("I", "ı").lower() if isinstance(text, str) else ""
        for text in texts
    ]
def map_labels_to_numeric(datasets, group_key):
    """
    Map labels to numeric values for all datasets in a group and optionally preprocess text for ColBERT.

    Args:
        datasets (list): List of datasets with file names, text columns, label columns, and paths.
        group_key (str): Key representing the dataset group (e.g., "news" or "sentiment").

    Returns:
        dict: Mapping table for the group.
    """
    mappings = {"news": {}, "sentiment": {}}
    all_labels = set()

    for file_name, text_column, label_column, path in datasets:
        file_path = os.path.join(str(path), str(file_name))
        try:
            # Load the dataset
            data = pd.read_csv(file_path)

            # Preprocess text specifically for ColBERT
            if model_name == "ytu-ce-cosmos-ColBERT":
                log_timestamp(f"Applying ColBERT-specific preprocessing to {file_name}...")
                data[text_column] = preprocess_colbert_text(data[text_column].tolist())

            # Collect all unique labels
            all_labels.update(data[label_column].unique())

            # Save the preprocessed dataset back (optional but recommended)
            data.to_csv(file_path, index=False)

        except Exception as e:
            log_timestamp(f"Error reading or processing {file_name}: {e}")

    # Create a sorted mapping of labels to numeric values
    sorted_labels = sorted(all_labels)
    mappings[group_key] = {label: idx for idx, label in enumerate(sorted_labels)}

    # Validate mapping success
    if not mappings[group_key]:
        log_timestamp(f"Error: Label mapping failed for group {group_key}")
        raise ValueError(f"Mapping failed for group {group_key}")

    # Apply label mapping to each dataset
    for file_name, _, label_column, path in datasets:
        file_path = os.path.join(str(path), str(file_name))
        try:
            data = pd.read_csv(file_path)
            data[label_column] = data[label_column].map(mappings[group_key])
            data.to_csv(file_path, index=False)
        except Exception as e:
            log_timestamp(f"Error applying label mapping to {file_name}: {e}")

    return mappings[group_key]

# Define dataset groups
news_datasets = [
    ("news_train.csv", "HABERLER", "ETIKET", processed_path),
    ("news_test.csv", "HABERLER", "ETIKET", processed_path),
    ("news_GPT2-750m-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
    ("news_GPT2-750m-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
    ("news_VBART-Large-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
    ("news_VBART-Large-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
    ("news_VBART-XLarge-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
    ("news_VBART-XLarge-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
]

sentiment_datasets = [
    ("sentiment_train.csv", "review", "score", processed_path),
    ("sentiment_test.csv", "review", "score", processed_path),
    ("sentiment_GPT2-750m-augmented_3_records.csv", "review", "score", augmented_path),
    ("sentiment_GPT2-750m-augmented_5_records.csv", "review", "score", augmented_path),
    ("sentiment_VBART-Large-augmented_3_records.csv", "review", "score", augmented_path),
    ("sentiment_VBART-Large-augmented_5_records.csv", "review", "score", augmented_path),
    ("sentiment_VBART-XLarge-augmented_3_records.csv", "review", "score", augmented_path),
    ("sentiment_VBART-XLarge-augmented_5_records.csv", "review", "score", augmented_path),
]

# Apply mapping
news_mapping = map_labels_to_numeric(news_datasets, group_key="news")
sentiment_mapping = map_labels_to_numeric(sentiment_datasets, group_key="sentiment")

def adjust_batch_size():
    """
    Dynamically adjust batch size based on available GPU memory.

    Returns:
        int: Recommended batch size.
    """
    # Check GPU memory availability
    free_memory = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0

    # Adjust batch size based on free memory (8 GB GPU-specific logic)
    if free_memory > 6 * 1024 ** 3:  # More than 6GB free
        return 64
    elif free_memory > 4 * 1024 ** 3:  # Between 4GB and 6GB free
        return 32
    elif free_memory > 2 * 1024 ** 3:  # Between 2GB and 4GB free
        return 16
    else:
        return 8  # Default smaller batch size for low memory

# Generate embeddings
def generate_embeddings(texts, transformer_model, tokenizer):
    """
    Generate embeddings for a list of texts using a transformer model.

    Args:
        texts (list): List of input texts to generate embeddings for.
        transformer_model (torch.nn.Module): Pretrained transformer model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the transformer model.

    Returns:
        np.ndarray: Array containing all generated embeddings in float32 format.
    """
    embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model.to(device)  # Ensure transformer model is on the correct device
    transformer_model.eval()
    texts = preprocess_colbert_text(texts)
    batch_size = adjust_batch_size()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = [str(text) for text in texts[i:i + batch_size] if text.strip()]
            if not batch_texts:
                continue
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = transformer_model(**inputs)

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Explicitly cast to float32
                batch_embeddings = outputs.pooler_output.to(torch.float32).cpu().numpy()
            else:
                # Handle embeddings from the last hidden state if pooler_output is unavailable
                batch_embeddings = outputs.last_hidden_state[:, 0, :].to(torch.float32).cpu().numpy()

            embeddings.append(batch_embeddings)

    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0,), dtype=np.float32)

# Train ML model
def train_ml_model(embeddings, labels, model_type):
    """
    Train an ML model with embeddings and labels.

    Args:
        embeddings (torch.Tensor): Input embeddings for training.
        labels (list): Corresponding labels.
        model_type (str): Type of ML model ("xgboost", "lightgbm", "adaboost", "svm", or "random_forest").

    Returns:
        model: Trained ML model.
    """
    # Ensure embeddings and labels sizes match
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Mismatch between embeddings ({len(embeddings)}) and labels ({len(labels)})."
        )

    # Convert embeddings to float32 (if needed)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy().astype(np.float32)

    # Initialize the model
    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(device="gpu", verbose=-1)
    elif model_type == "adaboost":
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=1.0
        )
    elif model_type == "svm":
        model = SVC(kernel="linear", probability=True, max_iter=1000)  # Linear kernel and probability scores
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    log_timestamp(f"Training {model_type}...")
    model.fit(embeddings, labels)
    log_timestamp(f"Training completed for {model_type} model.")
    return model

# Evaluate model (numerical labels expected)
def evaluate_model(model, embeddings, labels, dataset_name):
    """
    Evaluate the model and calculate metrics using numeric labels.
    """
    log_timestamp(f"Evaluating model on dataset: {dataset_name}")
    predictions = model.predict(embeddings)

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, average="weighted"),
        "Recall": recall_score(labels, predictions, average="weighted"),
        "F1-Score": f1_score(labels, predictions, average="weighted"),
    }
    log_timestamp(f"{dataset_name} - Metrics: {metrics}")
    return metrics, predictions

# Combine predictions with majority voting
def combine_predictions_with_borda_count(models_predictions, valid_labels):
    """
    Combines predictions using the Borda count method.

    Args:
        models_predictions (list of lists): Rankings from each model.
        valid_labels (set): Valid labels for the task.

    Returns:
        list: Combined predictions using Borda count.
    """
    if not models_predictions:
        raise ValueError("Models' predictions cannot be empty.")

    num_predictions = len(models_predictions[0])

    # Validate that all models have rankings for the same number of samples
    if not all(len(preds) == num_predictions for preds in models_predictions):
        raise ValueError("All models must have rankings for the same number of samples.")

    combined_preds = []
    for idx in range(num_predictions):
        # Gather rankings from all models for this specific sample
        rankings = [model_preds[idx] for model_preds in models_predictions]

        # Validate rankings
        if not all(isinstance(ranking, list) for ranking in rankings):
            raise ValueError(f"Expected rankings to be lists, but found {type(rankings)}")

        # Borda count aggregation
        borda_scores = defaultdict(int)
        for ranking in rankings:
            for rank, label in enumerate(ranking):
                borda_scores[label] += (len(ranking) - rank - 1)

        # Select the label with the highest Borda score
        if borda_scores:
            combined_preds.append(max(borda_scores, key=borda_scores.get))
        else:
            combined_preds.append(None)  # Handle cases with no valid predictions

    return combined_preds

# Evaluate ensemble predictions
def evaluate_ensemble_predictions(original_labels, ensemble_preds, dataset_name):
    """
    Evaluates ensemble predictions against true labels.

    Args:
        original_labels: True labels for the original dataset.
        ensemble_preds: Combined predictions after weighted majority voting.
        dataset_name: Name of the dataset for logging.

    Returns:
        dict: Metrics for ensemble predictions.
    """
    # Filter valid predictions
    valid_indices = [i for i, pred in enumerate(ensemble_preds) if pred is not None]
    filtered_original_labels = [original_labels[i] for i in valid_indices]
    filtered_ensemble_preds = [ensemble_preds[i] for i in valid_indices]

    if not valid_indices:
        return {"Accuracy": float('nan'), "Precision": float('nan'), "Recall": float('nan'), "F1-Score": float('nan')}

    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(filtered_original_labels, filtered_ensemble_preds),
        "Precision": precision_score(filtered_original_labels, filtered_ensemble_preds, average="weighted"),
        "Recall": recall_score(filtered_original_labels, filtered_ensemble_preds, average="weighted"),
        "F1-Score": f1_score(filtered_original_labels, filtered_ensemble_preds, average="weighted"),
    }

    return metrics

# Save results
def save_results_table(results, results_path):
    """
    Save results to a single CSV table with the required structure.

    Args:
        results (list): List of dictionaries with results.
        results_path (str): Path to save the CSV file.
    """
    log_timestamp("Saving results to CSV...")
    results_df = pd.DataFrame(results)
    output_path = os.path.join(results_path, f"performance_results_{model_name}.csv")
    try:
        results_df.to_csv(output_path, index=False)
        log_timestamp(f"Results saved successfully: {output_path}")
    except Exception as e:
        log_timestamp(f"Error saving results: {e}")
        raise

# Preprocess datasets (labels are assumed to be numeric after mapping)
def preprocess_datasets(datasets):
    """
    Preprocess datasets without remapping labels since map_labels_to_numeric is applied earlier.
    """
    log_timestamp("Preprocessing datasets and ensuring labels are numeric.")
    preprocessed_datasets = []

    # Validate labels are numeric
    for file_name, text_column, label_col, path in datasets:
        file_path = os.path.join(str(path), str(file_name))
        data = pd.read_csv(file_path)

        if not pd.api.types.is_numeric_dtype(data[label_col]):
            raise ValueError(f"Labels in {file_name} are not numeric. Ensure map_labels_to_numeric was applied.")

        preprocessed_datasets.append((file_name, text_column, label_col, path, data))

    log_timestamp("All datasets validated and preprocessed.")
    return preprocessed_datasets

def process_ensemble_results(datasets_group, tokenizer, transformer_model, train_dataset, device=None):
    """
    Process datasets and generate ensemble results for each original and augmented dataset.

    Args:
        datasets_group (list): List of datasets with paths and column names.
        tokenizer: Tokenizer for the transformer model.
        transformer_model: Transformer model for generating embeddings.
        train_dataset (tuple): The training dataset tuple (file_name, text_column, label_column, path).
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.

    Returns:
        list: Results for the datasets.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_model.to(device)
    results = []

    # Load and preprocess training data
    try:
        train_file_path = os.path.join(str(train_dataset[3]), str(train_dataset[0]))
        log_timestamp(f"Loading training dataset from: {train_file_path}")
        train_data_df = pd.read_csv(train_file_path)
        train_texts = train_data_df[train_dataset[1]].tolist()
        train_labels = train_data_df[train_dataset[2]].tolist()
        train_embeddings = generate_embeddings(train_texts, transformer_model, tokenizer)
    except Exception as e:
        log_timestamp(f"Error loading or processing training dataset: {e}")
        raise

    # Process each dataset in the group
    for dataset in datasets_group:
        dataset_name, text_column, label_column, path = dataset[:4]
        dataset_type = "Original" if "test.csv" in dataset_name else "Augmented"
        augmentation_size = (
            "3" if "3_records" in dataset_name else "5" if "5_records" in dataset_name else "N/A"
        )

        try:
            file_path = os.path.join(str(path), str(dataset_name))
            log_timestamp(f"Loading dataset from: {file_path}")
            data = pd.read_csv(file_path)
            texts = data[text_column].tolist()
            labels = data[label_column].tolist()
        except Exception as e:
            log_timestamp(f"Error loading or processing dataset {dataset_name}: {e}")
            raise

        log_timestamp(f"Processing {dataset_name} ({dataset_type}, Augmentation: {augmentation_size})")

        # Generate embeddings for the dataset
        embeddings = generate_embeddings(texts, transformer_model, tokenizer)

        # Train ML models
        ml_models = [
            train_ml_model(torch.tensor(train_embeddings), train_labels, model_type)
            for model_type in ["xgboost", "lightgbm", "adaboost", "svm", "random_forest"]
        ]

        # Evaluate models and collect predictions
        all_predictions = []
        for model, model_name in zip(ml_models, ["xgboost", "lightgbm", "adaboost", "svm", "random_forest"]):
            metrics, predictions = evaluate_model(model, torch.tensor(embeddings), labels, dataset_name)

            # Get probability scores and generate rankings (if applicable)
            prob_scores = model.predict_proba(embeddings)
            rankings = [np.argsort(prob)[::-1].tolist() for prob in prob_scores]  # Convert probabilities to rankings

            all_predictions.append(rankings)  # Append rankings, not scalar predictions
            log_timestamp(f"{dataset_name} ({model_name}) - Metrics: {metrics}")
            print(f"{dataset_name} ({model_name}) - Metrics: {metrics}")
            results.append({
                "Dataset": dataset_name,
                "Dataset Type": dataset_type,
                "Augmentation Size": augmentation_size,
                "ML Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"]
            })

        # Compute ensemble results using Borda count
        ensemble_preds = combine_predictions_with_borda_count(all_predictions, set(labels))
        ensemble_metrics = evaluate_ensemble_predictions(labels, ensemble_preds, dataset_name)

        # Log and save ensemble metrics here only
        log_timestamp(f"{dataset_name} - Ensemble Metrics: {ensemble_metrics}")
        results.append({
            "Dataset": dataset_name,
            "Dataset Type": dataset_type,
            "Augmentation Size": augmentation_size,
            "ML Model": "Ensemble (Borda)",
            "Accuracy": ensemble_metrics["Accuracy"],
            "Precision": ensemble_metrics["Precision"],
            "Recall": ensemble_metrics["Recall"],
            "F1-Score": ensemble_metrics["F1-Score"]
        })
    return results

def unified_process_with_combined_predictions():
    """
    Unified process for generating embeddings, training models, evaluating predictions,
    and processing ensembles with weighted majority voting.
    """
    log_timestamp("Starting the process with combined predictions")

    # Define datasets
    datasets = [
        ("news_train.csv", "HABERLER", "ETIKET", processed_path),
        ("news_test.csv", "HABERLER", "ETIKET", processed_path),
        ("sentiment_train.csv", "review", "score", processed_path),
        ("sentiment_test.csv", "review", "score", processed_path),
        ("news_GPT2-750m-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_GPT2-750m-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-Large-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-Large-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-XLarge-augmented_3_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("news_VBART-XLarge-augmented_5_records.csv", "HABERLER", "ETIKET", augmented_path),
        ("sentiment_GPT2-750m-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_GPT2-750m-augmented_5_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-Large-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-Large-augmented_5_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-XLarge-augmented_3_records.csv", "review", "score", augmented_path),
        ("sentiment_VBART-XLarge-augmented_5_records.csv", "review", "score", augmented_path),
    ]

    # Group datasets by type
    original_news_datasets = [d for d in datasets if "news_test" in d[0]]
    augmented_news_datasets = [d for d in datasets if "news" in d[0] and "augmented" in d[0]]

    original_sentiment_datasets = [d for d in datasets if "sentiment_test" in d[0]]
    augmented_sentiment_datasets = [d for d in datasets if "sentiment" in d[0] and "augmented" in d[0]]

    # Map labels to numeric values
    log_timestamp("Mapping labels to numeric values...")
    news_mapping = map_labels_to_numeric(original_news_datasets + augmented_news_datasets, group_key="news")
    sentiment_mapping = map_labels_to_numeric(original_sentiment_datasets + augmented_sentiment_datasets, group_key="sentiment")

    # Preprocess datasets
    log_timestamp("Preprocessing datasets...")
    news_preprocessed = preprocess_datasets(original_news_datasets + augmented_news_datasets)
    sentiment_preprocessed = preprocess_datasets(original_sentiment_datasets + augmented_sentiment_datasets)

    # Load tokenizer and model
    log_timestamp("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False, trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

    # Process news datasets (original and augmented)
    log_timestamp("Processing news datasets...")
    news_results = []
    for dataset in news_preprocessed:
        news_results.extend(process_ensemble_results([dataset], tokenizer, model, train_dataset=("news_train.csv", "HABERLER", "ETIKET", processed_path)))

    # Process sentiment datasets (original and augmented)
    log_timestamp("Processing sentiment datasets...")
    sentiment_results = []
    for dataset in sentiment_preprocessed:
        sentiment_results.extend(process_ensemble_results([dataset], tokenizer, model, train_dataset=("sentiment_train.csv", "review", "score", processed_path)))

    # Save results
    log_timestamp("Saving results to CSV...")
    results = news_results + sentiment_results
    save_results_table(results, results_path)
    log_timestamp("Process with combined predictions completed.")

if __name__ == "__main__":
    unified_process_with_combined_predictions()