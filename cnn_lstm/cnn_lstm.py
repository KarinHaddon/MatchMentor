"""Code for the CNN-LSTM model."""

import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor, bfloat16, nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from torchvision.models import DenseNet201_Weights, ResNet34_Weights
from tqdm import tqdm

# Constants
batch_sz = 8  # local: 4
n_classes = 4
frame_shape = (3, 224, 224)  # (n_channels, height, width)
n_channels, height, width = frame_shape[0], frame_shape[1], frame_shape[2]
seq_len = 10  # number of frames in each sequence


class CNNLSTM(nn.Module):
    """CNN-LSTM for video event classification."""
    def __init__(
            self,
            cnn_model: nn.Module,  # pre-trained CNN model
            lstm_input_sz: int,  # size of input features expected by LSTM
            lstm_hidden_sz: int,  # number of neurons in each lstm hidden layer
            lstm_n_layers: int,  # number of lstm layers
            n_classes: int = n_classes,  # number of output classes
            dropout: float = 0.15,  # dropout rate
        ):
        super().__init__()
        n_ch, w, h = 3, 224, 224  # input frame size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # <s CNN
        self.cnn = cnn_model  # pre-trained CNN model
        self.cnn = self.cnn.to(device, dtype=bfloat16)  # convert to bfloat16 and move to device
        self.cnn.classifier = nn.Identity()  # remove classification head, keep feature extractor
        # Dynamically compute CNN output size
        with torch.no_grad():
            # Simulate a single frame input
            dummy_input = torch.zeros(1, n_ch, w, h, dtype=bfloat16, device=device)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.shape[-1]
        self.cnn2lstm = nn.Linear(  # match CNN out to LSTM in
            self.cnn_output_size,
            lstm_input_sz,
            device=device,
            dtype=bfloat16
        )
        self.cnn_dropout = nn.Dropout(dropout)
        # /s>

        # <s LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_sz,
            hidden_size=lstm_hidden_sz,
            num_layers=lstm_n_layers,
            batch_first=True,
            dropout=dropout,
            dtype=bfloat16,
            device=device
        )
        self.lstm_dropout = nn.Dropout(dropout)
        # /s>

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_sz, n_classes, device=device, dtype=bfloat16)

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len n_channels height width"],  # sequence of frames
    ) -> Float[Tensor, "batch_sz num_classes"]:
        """Forward pass through CNN-LSTM: predicts output classes from frames.

        Args:
            x: The sequences of frames.

        Returns:
            Output tensor containing class probabilities for each sequence.

        Note:
            - The CNN processes each frame independently to extract spatial features.
            - The LSTM processes the sequence of CNN features to capture temporal patterns.
            - Only the final time step output from the LSTM is used for classification.
        """
        seq_len = x.size()[1]

        # Reshape for CNN; extract cnn features for each frame
        cnn_features = []
        for t in range(seq_len):
            frame_features = self.cnn(x[:, t, :, :, :])
            frame_features = self.cnn_dropout(frame_features)
            reduced_features = self.cnn2lstm(frame_features)  # cnn2lstm dimensionality matching
            cnn_features.append(reduced_features)
        cnn_features = torch.stack(cnn_features, dim=1)  # -> (batch_sz, seq_len, feature_sz)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # -> (batch_sz, seq_len, lstm_hidden_sz)
        lstm_out = self.lstm_dropout(lstm_out)

        # Get the last time step's output for classification
        output = self.fc(lstm_out[:, -1, :])  # -> (batch_sz, num_classes)

        return output

    def weights_init(self):
        """Initialize the weights of the CNN and LSTM layers using Xavier init."""
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear | nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name or 'weight_hh' in param_name:
                       nn. init.xavier_uniform_(param.data)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param.data)


def train(
    model: nn.Module,  # model to train
    train_loader: DataLoader,  # training data loader
    val_loader: DataLoader,  # validation data loader
    optimizer: optim.Optimizer,  # optimization algorithm
    max_epochs: int = 2,  # maximum number of epochs to train for
    max_batches: int = 100_000,  # maximum number of batches to train for
    val_every: int = 100,  # validate every n batch
    val_iter: int = 5,  # number of batches on val_loader to run and avg for val loss
    patience_thresh: int = 500,  # consecutive batches w/ no val loss decrease for early stop
) -> tuple[torch.Tensor, list, list, list]:  # -> final loss, train losses, val losses, batch_times
    """Trains the model, returns loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce_loss = nn.BCEWithLogitsLoss()  # binary cross entropy loss w/ sigmoided vals for stability

    # <s Nested helper functions to make `train` more readable.
    def print_losses(epoch, batch_i, train_losses_avg, val_losses_avg):
        """Print current average losses."""
        print(
            f"Epoch {epoch + 1}: Batch {batch_i + 1}:  "
            f"Loss = {train_losses_avg[-1]:.3f}, Val Loss = {val_losses_avg[-1]:.3f}"
        )

    @torch.no_grad()
    def estimate_losses(
        model: nn.Module,
        val_loader: DataLoader,
        val_losses: list[float],
        val_losses_avg: list[float],
        train_losses: list[float],
        train_losses_avg: list[float],
    ):
        """Estimate losses on val_loader, and return val loss and train loss avg."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        for val_i, (x_val, y_val) in enumerate(val_loader):
            logits = model(x_val.to(device))
            val_loss = bce_loss(logits, y_val.to(device))
            val_losses.append(val_loss.item())
            if val_i >= (val_iter - 1):
                break
        val_losses_avg.append(np.mean(val_losses[-val_iter:]))  # type: ignore
        train_losses_avg.append(np.mean(train_losses[-val_every:]))  # type: ignore
        model.train()
    # /s>

    # <s Trackers
    _batch_sz, n_batches = train_loader.batch_size, len(train_loader)
    batch_lim = min(max_batches, n_batches * max_epochs)
    train_losses, val_losses, train_losses_avg, val_losses_avg, batch_times = [], [], [], [], []
    best_val_loss = float("inf")
    patience_ct = 0
    # /s>

    # <s Training loop
    for epoch in range(max_epochs):
        pbar = tqdm(enumerate(train_loader), total=batch_lim, desc="Batch progression")
        for batch_i, (x_train, y_train) in pbar:
            start_time = time.time()
            # <ss Model training.
            optimizer.zero_grad()
            # Forward pass
            logits = model(x_train.to(device))  # -> (batch_sz, num_classes)
            loss = bce_loss(logits, y_train.to(device))  # targets are float for bce loss
            # Backward pass and params update step
            loss.backward()
            apply_gradient_centralization(optimizer)
            optimizer.step()
            train_losses.append(loss.item())
            batch_times.append(time.time() - start_time)
            # /ss>
            # <ss Model validation.
            if val_every and batch_i % val_every == 0:
                # Estimate and print losses.
                estimate_losses(
                    model,
                    val_loader,
                    val_losses,
                    val_losses_avg,
                    train_losses,
                    train_losses_avg,
                )
                print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                pbar.set_postfix_str(f"Total Batch {(batch_i + 1) * (epoch + 1)} / {batch_lim}")
                # Patience check for early stopping.
                patience_ct = (
                    0 if val_losses_avg[-1] < best_val_loss else patience_ct + val_every
                )
                best_val_loss = min(best_val_loss, val_losses_avg[-1])
                if patience_ct >= patience_thresh:
                    print("Early stopping.")
                    print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                    return loss, train_losses_avg, val_losses_avg, batch_times
            # Max batch check.
            if (batch_i + 1) * (epoch + 1) >= max_batches:
                print("Finished training:")
                print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                return loss, train_losses_avg, val_losses_avg, batch_times
            # /ss>
        # /s>

    print("Finished training:")
    print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)  # type: ignore
    return loss, train_losses_avg, val_losses_avg, batch_times  # type: ignore


def apply_gradient_centralization(optimizer):
    """Applies gradient centralization to the optimizer.

    This function should be called before optimizer.step() in the training loop.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                # Compute the mean of the gradient
                grad_mean = param.grad.data.mean(
                    dim=tuple(range(1, len(param.grad.shape))), keepdim=True
                )
                # Centralize the gradient
                param.grad.data -= grad_mean


def load_data(
    seq_len: int,  # number of frames in each sequence
    labels_path: Path,  # path to the labels CSV file
    frames_path: Path,  # path to the directory containing frames
    height=224,  # image height
    width=224  # image width
) -> (
    tuple[Float[Tensor, "n_sequences seq_len n_channels height width"],  # X
          Float[Tensor, "n_sequences num_classes"]]  # Y
):
    """Returns inputs and target outputs for the CNN-LSTM model."""
    # Load labels and set target outpouts
    labels_df = pd.read_csv(labels_path)
    label_cols = ["Possession", "Passing", "InPlay", "Goals"]
    Y = torch.from_numpy(labels_df[label_cols].values).float()

    # Load frames and set input
    frames_dir = Path(frames_path)
    frame_files = sorted(frames_dir.glob("*.png"), key=lambda x: int(x.stem.split('_')[1]))
    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    X_init = torch.stack(
        [transform(Image.open(f).convert("RGB")) for f in frame_files]
    ).to(dtype=torch.bfloat16)
    # Create input sequences
    n_sequences = len(frame_files) - seq_len + 1
    X = torch.stack([X_init[s : s + seq_len] for s in range(n_sequences)])

    # Align Y with X
    Y = Y[:n_sequences]

    return X, Y


def objective(
    trial: optuna.Trial,  # Optuna trial object: one model training run
    labels_path: Path,  # path to the labels CSV file
    frames_path: Path,  # path to the directory containing frames
    cnn: nn.Module,  # pre-trained cnn
    n_workers: int = 4,  # number of workers for data loaders
) -> float:  # -> final validation loss
    """Optuna objective function."""
    # 1. Sample Hyperparameters
    seq_len = trial.suggest_categorical("seq_len", [15, 20, 25, 30])  # (for local: [10, 15])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    beta1 = trial.suggest_float("beta1", 0.9, 0.99)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # 2. Define Model
    lstm_input_sz = 512  # Fixed; adjust if necessary
    lstm_hidden_sz = 512  # Fixed; adjust if necessary
    model = CNNLSTM(
        cnn_model=cnn,
        lstm_input_sz=lstm_input_sz,
        lstm_hidden_sz=lstm_hidden_sz,
        lstm_n_layers=2,
        n_classes=n_classes,
        dropout=dropout,
    )
    model.weights_init()

    # 3. Define Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            fused=True,
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            fused=True,
        )
    else:
        raise ValueError("Unsupported optimizer type.")

    # 4. Prepare Data Loaders with Specified Sequence Length
    X, Y = load_data(seq_len, labels_path, frames_path)
    dataset = TensorDataset(X, Y)
    train_ratio, val_ratio, test_ratio = 0.85, 0.1, 0.05
    train_data, val_data, test_data = random_split(dataset, [train_ratio, val_ratio, test_ratio])

    train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_sz, shuffle=True, num_workers=n_workers)
    _test_loader = DataLoader(test_data, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    # 5. Train the Model
    _loss, _train_losses_avg, val_losses_avg, _batch_times = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_epochs=1,
    )

    return val_losses_avg[-1]


# Run optuna study
if __name__ == "__main__":
    # Set up params
    # local: labels_path = Path(r"C:\Users\jai\MatchMentor\cnn_lstm\data\LSTMLabels.csv")
    # local: frames_path = Path(r"C:\Users\jai\MatchMentor\cnn_lstm\data\frames")
    labels_path = Path("~").expanduser() / "MatchMentorData" / "LSTMLabels.csv"
    frames_path = Path("~").expanduser() / "MatchMentorData" / "frames"
    cnn = models.densenet201(weights=DenseNet201_Weights.DEFAULT)

    # Set up and run optuna study
    study = optuna.create_study(direction="minimize", study_name="CNN-LSTM")
    study.optimize(
        lambda trial: objective(
            trial,
            labels_path=labels_path,
            frames_path=frames_path,
            cnn=cnn,
        ),
        n_trials=50,
    )

    # View some results
    # Print info on all trials and best trial
    print(f"Number of finished trials: {len(study.trials)} \n")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best hyperparameters: \n\n")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    print(study.best_params)
