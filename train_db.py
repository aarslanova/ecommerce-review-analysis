import argparse
import mlflow
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertModel


parser = argparse.ArgumentParser(description='Training a BERT model for regression.')

parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size for training and evaluation')
parser.add_argument('--LR', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--NUM_EPOCHS', type=int, default=3, help='Number of epochs to train for')
parser.add_argument('--NUM_WORKERS', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--TRAIN_SIZE', type=float, default=0.8, help='Proportion of data to use for training')
parser.add_argument('--BERT_MODEL_NAME', type=str, default="distilbert-base-uncased", help="Name of the DistilBERT model from huggingface")

args = parser.parse_args()


class ReviewDataset(Dataset):
    def __init__(
        self,
        tokenizer: DistilBertModel,
        texts: list[str],
        scores: list[int],
        max_len: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.scores = scores
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(
        self, idx
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        text = self.texts[idx]
        score = self.scores[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return (encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()), torch.tensor(
            score, dtype=torch.float
        )


class RegressionHead(nn.Module):
    def __init__(self, distilbert_model: nn.Module):
        super().__init__()
        self.distilbert_model = distilbert_model
        self.regressor = nn.Sequential(
            nn.Linear(distilbert_model.config.dim, distilbert_model.config.dim),
            nn.Dropout(0.2),
            nn.Linear(distilbert_model.config.dim, 1),
        )

        for param in self.distilbert_model.parameters():
            param.requires_grad = False

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input_ids, attention_mask = inputs
        outputs = self.distilbert_model(
            input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.last_hidden_state[:, 0])


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for inputs, target in tqdm(dataloader, desc='Training', leave=False):
        inputs = [i.to(device) for i in inputs]
        target = target.to(device).float()

        optimizer.zero_grad()
        predictions = model(inputs).squeeze()
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def test_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    with torch.inference_mode():
        total_loss = 0
        progress_bar = tqdm(dataloader, desc='Testing', leave=False)
        for inputs, target in dataloader:
            inputs = [i.to(device) for i in inputs]
            target = target.to(device).float()

            predictions = model(inputs).squeeze()
            loss = criterion(predictions, target)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    BATCH_SIZE: int = args.BATCH_SIZE
    BERT_MODEL_NAME: str = args.BERT_MODEL_NAME
    DATASET_PATH: str = "data/reviews.parquet"
    DEVICE: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    LR: float = args.LR
    NUM_EPOCHS: int = args.NUM_EPOCHS
    NUM_WORKERS: int = args.NUM_WORKERS
    TRAIN_SIZE: float = args.TRAIN_SIZE

    df = pd.read_parquet(DATASET_PATH)
    df = df.dropna(subset=["review/text", "review/score"])
    texts = df["review/text"]
    scores = df["review/score"]

    distillbert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    distillbert_model = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
    distillbert_model.to(DEVICE)

    model = RegressionHead(distillbert_model)
    model.to(DEVICE)

    dataset = ReviewDataset(distillbert_tokenizer, texts, scores)
    train_size_len = int(TRAIN_SIZE * len(dataset))
    test_size_len = len(dataset) - train_size_len

    train_dataset, test_dataset = random_split(dataset, [train_size_len, test_size_len])
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    mlflow.set_experiment("BERT_Regression_Reviews")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "TRAIN_SIZE": TRAIN_SIZE,
                "BATCH_SIZE": BATCH_SIZE,
                "NUM_EPOCHS": NUM_EPOCHS,
                "NUM_WORKERS": NUM_WORKERS,
                "BERT_MODEL_NAME": BERT_MODEL_NAME,
                "LR": LR
            }
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train_loss = train_model(
                model, train_dataloader, optimizer, criterion, DEVICE
            )
            test_loss = test_model(model, test_dataloader, criterion, DEVICE)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f} - Test loss: {test_loss:.4f}"
            )

        mlflow.pytorch.log_model(model, "model")
