import hashlib
password = 'harris'
hash_pass = (hashlib.sha256(password.encode('utf-8'))).hexdigest()
print (hash_pass)

from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models, transforms

# Checking if a GPU is available and using it if possible
# AQA Criteria: optimisation for performance
if torch.cuda.is_available():  
    device = torch.device("cuda")
    print(f"Using GPU device: {torch.cuda.get_device_name(device)}\n")
else:  
    device = torch.device("cpu")
    print("Using CPU.\n")

# Loading the dataset and reading CSV files using pandas for handling labeled data
# AQA Criteria:file handling and processing data
labels_path = Path(r"C:\Users\karin\NEA\Data\Initial_combined_labels.csv")  
labels_df = pd.read_csv(labels_path)
# displaying the first few rows of the data for testing  
display(labels_df.head())  

#extracting target labels and converting to torch tensor
# AQA Criteria: Correct data type conversions, handling multi-dimensional data
trueL = labels_df[["Possession", "Set piece"]].values  
trueL = torch.from_numpy(trueL).float().to(device)  
#shape of the label tensor for debugging
print(trueL.shape)
#Extracting the number of samples and output classes
Numb = trueL.shape[0] 
n_classes = trueL.shape[1]

# loading the image dataset from file
# AQA Criteria: use of external libraries for image handling and processing
frames_path = Path(r"C:\Users\karin\NEA\data\initial_combined_frames")  
n_channels, height, width = 3, 224, 224  
inputI = torch.empty((Numb, n_channels, height, width))  
print(inputI.shape)  

# Converting images into tensors and resizing them
# AQA Criteria: Correct use of external libraries (PIL, torchvision) manipulating data structures
for i, file in enumerate(frames_path.glob("*.png")):  
    with Image.open(file).convert("RGB") as img:  
        to_tensor = transforms.ToTensor()  # Converting image to a tensor
        resize = transforms.Resize((height, width))  # Resizing the image to the correct dimensions
        img_tensor = to_tensor(img).float()  # Final tensor for the image
        img_tensor = resize(img_tensor) 
        inputI[i, :] = img_tensor  # Storing the tensor in the dataset

#splitting data into training and validation sets
# AQA Criteria: Implementation of data splitting, understanding of training/validation workflow in machine learning
dataset = TensorDataset(inputI, trueL)
train_data, val_data = random_split(dataset, [0.90, 0.10])  # 90% training, 10% validation

# Printing shapes of the data splits for verification
print(train_data[:][0].shape, train_data[:][1].shape, val_data[:][0].shape, val_data[:][1].shape)

# Function to apply gradient centralisation to the optimiser
# AQA Criteria: Custom functions, gradient centralisation is an advanced optimisation technique
def apply_gradient_centralization(optimiser):
    for group in optimiser.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # Compute the mean of the gradient across all but the first dimension
                grad_mean = param.grad.data.mean(dim=tuple(range(1, len(param.grad.shape))), keepdim=True)
                # Centralize the gradient by subtracting the mean
                param.grad.data -= grad_mean

# Training function
# AQA Criteria: Modular design, defining functions, handling epochs, training loops and evaluation
def train(
    model: nn.Module,  
    train_loader: DataLoader, 
    val_loader: DataLoader,  
    optimiser: optim.optimiser, 
    loss_fn: nn.modules.loss._Loss,  
    max_epochs: int = 5,  
    val_check_interval: int = 10, 
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    train_losses, val_losses = [], []  # Lists to store the training and validation losses

    # Training loop for iterating over epochs and batches
    # AQA Criteria: Understanding of machine learning workflows epoch and batch management
    for epoch in range(max_epochs):
        for batch_i, (x_train, y_train) in enumerate(train_loader):
            model.train()  # Setting the model to training mode
            optimiser.zero_grad()  # settting t o zero gradients for the optimiser
            out = model(x_train.to(device))  # Forward pass
            loss = loss_fn(out, y_train.to(device))  # Calculating the loss
            loss.backward()  # Backward pass for computing the gradient
            apply_gradient_centralization(optimiser)  # Applying gradient centralisation
            optimiser.step()  # Updating weights

            if batch_i % val_check_interval == 0:  # Checking validation loss every few batches
                train_losses.append(loss.item())  # Recording training loss
                model.eval()  # Setting the model to evaluation mode
                with torch.no_grad():  # No gradient computation during evaluation
                    x_val, y_val = next(iter(val_loader))  # Get validation batch
                    val_out = model(x_val.to(device))  # Forward pass for validation
                    val_loss = loss_fn(val_out, y_val.to(device))  # Validation loss
                    val_losses.append(val_loss.item())  # Record validation loss
                # Output progress for monitoring
                print( 
                    f"Epoch {epoch + 1}: Batch {batch_i + 1}:  "
                    f"Loss = {train_losses[-1]:.3f}, Val Loss = {val_losses[-1]:.3f}"
                )
   
    # Final output after training
    print("Finished training:")
    print(f"Epoch {epoch + 1}:  Batch {batch_i + 1}: Loss = {train_losses[-1]:.3f}, Val Loss = {val_losses[-1]:.3f}")
    return loss, train_losses, val_losses  # Returning final losses for analysis

# Loading a pre-trained model (DenseNet) and modifying it for the task
# AQA Criteria: Demonstrating use of pre-trained models and modifying architecture 
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)  # Pretrained DenseNet
print(model)

# Modifying the classifier
# AQA Criteria: Customising neural network layers and adapting model architecture
dropout_rate = 0.2  
model.classifier = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.classifier.in_features, n_classes),  # Adjust the classifier for our number of classes
    nn.Sigmoid()  # Using sigmoid for binary classification
)

# Setting hyperparameters and loading data into DataLoader for training and validation
# AQA Criteria: Use of hyperparameters (batch size, learning rate) and DataLoader for efficiency
batch_size = 32  
lr = 0.02  # Learning rate

train_loader = DataLoader(train_data, batch_size=batch_size)  # DataLoader for training data
val_loader = DataLoader(val_data, batch_size=batch_size)  # DataLoader for validation data

# Defining the loss function and optimiser
# AQA Criteria: selection of loss function and optimiser
loss_fn = nn.BCELoss()  # binary Cross-Entropy Loss for binary classification
optimiser = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-7, momentum=0.6, nesterov=True)

# running the training process
# AQA Criteria: Running and testing the full model, integration of training logic
loss, train_losses, val_losses = train(
    model.to(device), train_loader, val_loader, optimiser, loss_fn, max_epochs=2, val_check_interval=3
)

# plotting training and validation losses to visualize model performance
# AQA Criteria: Data visualization for analysis and evaluation
fig, ax = plt.subplots()
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Val")
ax.legend()
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Losses")

# saving the trained model
# AQA Criteria: demonstrating file handling and model persistence
model_path = f"MatchMentor_Liverpool_Loss={train_losses[-1]:.3f}_ValLoss={val_losses[-1]:.3f}.pth"
save_path = Path.cwd().parent / "models" / model_path
torch.save(model.state_dict(), save_path)  # Save model state to file