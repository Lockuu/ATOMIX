import os.path

import yaml
from pathlib import Path
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from src.data_processing import preprocess_and_split_data
from datasets import load_dataset
from src.model_config import ModelConfig
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict

# Set up logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class DiseaseClassifier:
    def __init__(self, config_path="model_config.yaml"):
        # Load configuration
        self.config_path = Path(__file__).parent / config_path
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file missing at {self.config_path}")

        # Load YAML config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if not self.config or 'data' not in self.config:
            raise ValueError("Invalid config format! Missing 'data' section")

        train_path = Path(self.config['data']['train_path'])
        test_path = Path(self.config['data']['test_path'])

        if not train_path.exists() or not test_path.exists():
            preprocess_and_split_data(self.config)

        self.model_config = ModelConfig(**self.config["model"])
        self.training_config = self.config["training"]
        self.data_config = self.config["data"]

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        self.model = self._init_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _init_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            config=self.model_config.get_huggingface_config()
        )

    def load_data(self):
        """Load and preprocess dataset"""

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.training_config["max_seq_length"]
            )

        # Check if dataset paths exist and are non-empty
        train_path = Path(self.data_config["train_path"])
        test_path = Path(self.data_config["test_path"])

        if not train_path.exists() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"Train dataset file is missing or empty: {train_path}")
        if not test_path.exists() or test_path.stat().st_size == 0:
            raise FileNotFoundError(f"Test dataset file is missing or empty: {test_path}")

        # Load datasets using Hugging Face `datasets` library
        try:
            dataset = load_dataset("csv", data_files={
                "train": str(train_path),
                "test": str(test_path)
            }, keep_in_memory=True)
        except Exception as e:
            raise ValueError(f"Error loading datasets. Please check the CSV files. Error: {str(e)}")

        # Tokenization
        try:
            dataset = dataset.map(tokenize, batched=True)
        except Exception as e:
            raise ValueError(f"Error during tokenization: {str(e)}")

        return dataset

    def train(self):
        """Train the model manually without using Trainer/TrainingArguments"""
        dataset = self.load_data()

        # Set Hugging Face dataset formatting for pytorch tensors
        # Add 'token_type_ids' if tokenizer provides it
        columns = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in dataset["train"].column_names:
            columns.append("token_type_ids")

        dataset.set_format(type="torch", columns=columns)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        train_loader = DataLoader(train_dataset, batch_size=self.training_config["batch_size"], shuffle=True)
        eval_loader = DataLoader(test_dataset, batch_size=self.training_config["batch_size"])

        self.training_config["learning_rate"] = float(self.training_config["learning_rate"])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config["learning_rate"])
        num_epochs = self.training_config["epochs"]

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for batch in train_loader:
                # inputs are already tensors; just move to device except label
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

            # Evaluation after each epoch
            self.model.eval()
            eval_loss = 0
            eval_correct = 0
            eval_total = 0

            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                    labels = batch["label"].to(self.device)
                    outputs = self.model(**inputs)
                    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                    eval_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(outputs.logits, dim=1)
                    eval_correct += (preds == labels).sum().item()
                    eval_total += labels.size(0)

            eval_loss /= eval_total
            eval_accuracy = eval_correct / eval_total
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Eval loss: {eval_loss:.4f} - Eval accuracy: {eval_accuracy:.4f}")

            self.model.train()

        # Save model and tokenizer manually
        save_dir = "saved_models/best_model"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Training completed and model saved.")

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return {"accuracy": (predictions == labels).float().mean().item()}

    def predict(self, text):
        """Make prediction on new text"""
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.training_config["max_seq_length"],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        return {
            "prediction": torch.argmax(probs).item(),
            "probabilities": probs.cpu().numpy().tolist()[0]
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--text", help="Text to predict", default=None)
    args = parser.parse_args()

    classifier = DiseaseClassifier()

    if args.mode == "train":
        classifier.train()
    elif args.mode == "predict":
        if not args.text:
            raise ValueError("Please provide --text for prediction")
        result = classifier.predict(args.text)
        print(f"Prediction: {result['prediction']}")
        print(f"Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()