"""
Image Classification Neural Network for Dogs vs Cats Dataset.

This module implements a neural network for binary image classification,
classifying images as either cats (0) or dogs (1).
The model uses a convolutional neural network (CNN) approach for image feature extraction
followed by fully connected layers for classification.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import random

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data and File Configuration
TRAIN_DIR = 'data/train_set'
TEST_DIR = 'data/test_set'
MODEL_STATE_FILE_NAME = "best trained model.pth"  # Set to None to train from scratch
TEST_ONLY = True  # Whether to only test a pre-trained model (and not train it futher)
DATA_SPLIT_SEED = 12345

# Model Architecture Parameters
N_CLASSES = 2  # Binary classification: cats (0) vs dogs (1)
CNN_CHANNELS = (32, 64, 128)  # Convolutional layer channels
FC_HIDDEN_SIZES = (512, 256, 128)  # Fully connected layer sizes
OUT_LAYER_SIZE = N_CLASSES

# Training Parameters
BATCH_SIZE = 128
DROPOUT_AMOUNT = 0.3
INIT_LEARNING_RATE = 0.001
INIT_WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 10
MAX_EPOCHS = 10000
EPOCHS_PER_VALIDATION = 1
# EPOCHS_PER_SAVE_STATE = 10
PATIENCE_LIMIT = 10

# Image Processing Parameters
IMG_SIZE = 128  # Resize images to 128x128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization values
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Other Parameters
EPSILON = 1e-7  # Small constant to prevent division by zero
TRY_TO_OVERFIT = False  # Should be false normally
DEBUG = 0


class DogsVsCatsDataset(Dataset):
    """
    Custom Dataset class for Dogs vs Cats image data.
    
    Handles loading images, applying transforms, and extracting labels from filenames.
    """

    def __init__(self, image_paths, labels=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of paths to image files
            labels (list, optional): List of labels (0 for cat, 1 for dog)
            transform (callable, optional): Transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def create_data_transforms():
    """
    Create data augmentation and normalization transforms.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    return train_transform, val_transform


def load_image_data(train_split: float = 0.8):
    """
    Load Dogs vs Cats data from existing train/ and test/ folders.

    Args:
        data_dir (str): Base directory containing TRAIN_DIR and TEST_DIR subfolders.
        train_split (float): Fraction of the 'train' folder to use for training (rest for validation).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # --- 1) Collect and label all images in train_dir ---
    train_image_paths = []
    train_labels = []
    for img_path in glob.glob(os.path.join(TRAIN_DIR, '*.jpg')):
        train_image_paths.append(img_path)
        fname = os.path.basename(img_path)
        if fname.startswith('cat.'):
            train_labels.append(0)
        elif fname.startswith('dog.'):
            train_labels.append(1)
        else:
            raise ValueError(f"Unknown label for {fname}")

    # --- 2) Split into train / validation ---
    split_idx = int(len(train_image_paths) * train_split)
    # For reproducibility/shuffle you can shuffle with a fixed seed here if you like.
    combined = list(zip(train_image_paths, train_labels))
    random.seed(DATA_SPLIT_SEED)
    random.shuffle(combined)
    paths_shuf, labels_shuf = zip(*combined)
    X_train_paths = paths_shuf[:split_idx]
    y_train = labels_shuf[:split_idx]
    X_val_paths = paths_shuf[split_idx:]
    y_val = labels_shuf[split_idx:]

    # --- 3) Collect and label all images in test_dir ---
    test_image_paths = []
    test_labels = []
    for img_path in glob.glob(os.path.join(TEST_DIR, '*.jpg')):
        test_image_paths.append(img_path)
        fname = os.path.basename(img_path)
        if fname.startswith('cat.'):
            test_labels.append(0)
        elif fname.startswith('dog.'):
            test_labels.append(1)
        else:
            raise ValueError(f"Unknown label for {fname}")

    # --- 4) Create transforms & datasets/loaders ---
    train_transform, val_transform = create_data_transforms()

    train_ds = DogsVsCatsDataset(X_train_paths, list(y_train), train_transform)
    val_ds = DogsVsCatsDataset(X_val_paths, list(y_val), val_transform)
    test_ds = DogsVsCatsDataset(test_image_paths, test_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=10, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=10, pin_memory=True,
                            prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=10, pin_memory=True,
                             prefetch_factor=2, persistent_workers=True)

    return train_loader, val_loader, test_loader


# def load_image_data(data_dir: str):
#     """
#     Load and prepare Dogs vs Cats image data.
#
#     Args:
#         data_dir (str): Path to the directory containing train/ folder
#
#     Returns:
#         tuple: DataLoaders for train, validation, and test sets
#     """
#     train_dir = os.path.join(data_dir, TRAIN_DIR)
#
#     # Get all image paths and extract labels
#     image_paths = []
#     labels = []
#     print(train_dir)
#     for img_path in glob.glob(os.path.join(train_dir, '*.jpg'), recursive=True):
#         image_paths.append(img_path)
#         filename = os.path.basename(img_path)
#         # Extract label from filename (cat.1.jpg -> 0, dog.1.jpg -> 1)
#         if filename.startswith('cat.'):
#             labels.append(0)
#         elif filename.startswith('dog.'):
#             labels.append(1)
#         else:
#             print(f"Warning: Unknown label for file {filename}")
#
#     # Split data
#     X_train_paths, X_test_paths, y_train, y_test = train_test_split(
#         image_paths, labels, test_size=0.2, random_state=DATA_SPLIT_SEED, stratify=labels
#     )
#
#     X_train_paths, X_val_paths, y_train, y_val = train_test_split(
#         X_train_paths, y_train, test_size=0.25, random_state=DATA_SPLIT_SEED, stratify=y_train
#     )
#
#     # Create transforms
#     train_transform, val_transform = create_data_transforms()
#
#     # Create datasets
#     train_dataset = DogsVsCatsDataset(X_train_paths, y_train, train_transform)
#     val_dataset = DogsVsCatsDataset(X_val_paths, y_val, val_transform)
#     test_dataset = DogsVsCatsDataset(X_test_paths, y_test, val_transform)
#
#     # Create data loaders
#     # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=8,  # or more, depending on CPU cores
#         pin_memory=True,  # allocate in page-locked memory
#         prefetch_factor=2,  # number of batches to prefetch
#         persistent_workers=True  # keep workers alive across epochs
#     )
#     # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=8,  # or more, depending on CPU cores
#         pin_memory=True,  # allocate in page-locked memory
#         prefetch_factor=2,  # number of batches to prefetch
#         persistent_workers=True  # keep workers alive across epochs
#     )
#     # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=8,  # or more, depending on CPU cores
#         pin_memory=True,  # allocate in page-locked memory
#         prefetch_factor=2,  # number of batches to prefetch
#         persistent_workers=True  # keep workers alive across epochs
#     )
#
#     return train_loader, val_loader, test_loader


def con_mat(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate confusion matrix for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        torch.Tensor: 2x2 confusion matrix
    """
    cm = torch.zeros((N_CLASSES, N_CLASSES), device=DEVICE)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            cm[i, j] = ((y_true == i) & (y_pred == j)).sum()
    return cm


def calculate_metrics(confusion_matrix: torch.Tensor) -> dict:
    """
    Calculate precision, recall, and F1-score for each class.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        
    Returns:
        dict: Dictionary containing precision, recall, and F1-score for each class,
              plus macro-averaged F1 score
    """
    metrics = {}
    n_classes = confusion_matrix.shape[0]

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)

        metrics[f'class_{i}'] = {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

    metrics['macro_f1'] = sum(metrics[f'class_{i}']['f1'] for i in range(n_classes)) / n_classes
    return metrics


class DogsVsCatsCNN(nn.Module):
    """
    Convolutional Neural Network for Dogs vs Cats image classification.
    
    The network consists of convolutional layers for feature extraction
    followed by fully connected layers for classification.
    """

    def __init__(self, cnn_channels: tuple, fc_hidden_sizes: tuple):
        """
        Initialize the CNN.
        
        Args:
            cnn_channels: Tuple of CNN channel sizes
            fc_hidden_sizes: Tuple of fully connected layer sizes
        """
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB images

        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if not TRY_TO_OVERFIT else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(DROPOUT_AMOUNT) if not TRY_TO_OVERFIT else nn.Identity()
            ))
            in_channels = out_channels

        # Calculate flattened size after conv layers
        # After 3 MaxPool2d operations, spatial size is reduced by 2^3 = 8
        conv_output_size = cnn_channels[-1] * (IMG_SIZE // (2 ** len(cnn_channels))) ** 2

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_input_size = conv_output_size

        for fc_size in fc_hidden_sizes:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_input_size, fc_size),
                nn.BatchNorm1d(fc_size) if not TRY_TO_OVERFIT else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(DROPOUT_AMOUNT) if not TRY_TO_OVERFIT else nn.Identity()
            ))
            fc_input_size = fc_size

        # Output layer
        self.output_layer = nn.Linear(fc_hidden_sizes[-1], OUT_LAYER_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Output layer
        x = self.output_layer(x)

        return x


def train_model(model: DogsVsCatsCNN, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
    """
    Train the CNN model.
    
    Args:
        model: The CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INIT_LEARNING_RATE,
        weight_decay=INIT_WEIGHT_DECAY if not TRY_TO_OVERFIT else 0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=SCHEDULER_PATIENCE
    )

    best_val_loss = float('inf')
    best_val_macro_f1 = 0
    best_val_acc = 0
    patience_counter = 0

    # Create log files to help create graphs
    train_loss_txt = open("train_loss.txt", "w")
    val_loss_txt = open("val_loss.txt", "w")
    val_acc_txt = open("val_acc.txt", "w")
    train_loss_txt.write("Train Loss\n")
    val_loss_txt.write("Validation Loss\n")
    val_acc_txt.write("Validation Accuracy\n")

    for epoch in range(epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # images, labels = images.to(DEVICE), labels.to(DEVICE)
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss_total / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        if epoch % EPOCHS_PER_VALIDATION == 0:
            model.eval()
            val_loss_total = 0
            val_correct = 0
            val_total = 0
            all_val_labels = []
            all_val_predictions = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    val_loss_total += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_predictions.extend(predicted.cpu().numpy())

            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = val_correct / val_total

            # Calculate F1 scores
            val_labels_tensor = torch.tensor(all_val_labels, device=DEVICE)
            val_pred_tensor = torch.tensor(all_val_predictions, device=DEVICE)
            val_confusion_mat = con_mat(val_labels_tensor, val_pred_tensor)
            val_metrics = calculate_metrics(val_confusion_mat)
            val_macro_f1 = val_metrics['macro_f1']

            # Print progress
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.5f}, Train Acc = {train_acc:.5f}, "
                  f"Val Loss = {avg_val_loss:.5f}, Val Acc = {val_acc:.5f}, Val F1 = {val_macro_f1:.5f}")

            train_loss_txt.write(f"{avg_train_loss}\n")
            val_loss_txt.write(f"{avg_val_loss}\n")
            val_acc_txt.write(f"{val_acc}\n")

            # Save best model
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                save_model(model, f"best_model_f1_{val_macro_f1:.5f}.pth")
                patience_counter = 0
            elif val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, f"best_model_acc_{val_acc:.5f}.pth")
                patience_counter = 0
            elif avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # save_model(model, f"best_model_loss_{avg_val_loss:.5f}.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(avg_val_loss)

        # # Save checkpoint
        # if epoch % EPOCHS_PER_SAVE_STATE == 0 and epoch > 0:
        #     save_model(model, f"checkpoint_epoch_{epoch}.pth")

        # Early stopping check
        if patience_counter >= PATIENCE_LIMIT:
            print(f"Early stopping triggered ({PATIENCE_LIMIT} epochs without improvement)")
            break

    train_loss_txt.close()
    val_loss_txt.close()
    val_acc_txt.close()


def save_model(model: DogsVsCatsCNN, filename: str) -> None:
    """
    Save the model state to file.
    
    Args:
        model: Trained model to save
        filename: Filename for saving
    """
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{filename}")


def load_model(filename: str) -> DogsVsCatsCNN:
    """
    Load model from file.
    
    Args:
        filename: Filename for loading
        
    Returns:
        DogsVsCatsCNN: Loaded model
    """
    model = DogsVsCatsCNN(CNN_CHANNELS, FC_HIDDEN_SIZES)
    model.load_state_dict(torch.load(filename, weights_only=True))
    return model


def evaluate_model(model: DogsVsCatsCNN, test_loader: DataLoader) -> None:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model to evaluate
        test_loader: Test data loader
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_test_labels = []
    all_test_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    test_acc = test_correct / test_total

    # Calculate metrics
    test_labels_tensor = torch.tensor(all_test_labels, device=DEVICE)
    test_pred_tensor = torch.tensor(all_test_predictions, device=DEVICE)
    test_confusion_mat = con_mat(test_labels_tensor, test_pred_tensor)
    test_metrics = calculate_metrics(test_confusion_mat)

    # Print results
    print("\nModel Performance on Test Set:")
    print(f"Accuracy: {test_acc:.5f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.5f}")
    print(f"\nConfusion Matrix:")
    print(f"        Predicted")
    print(f"            Cat  Dog")
    print(f"Actual Cat  {int(test_confusion_mat[0, 0]):3d}  {int(test_confusion_mat[0, 1]):3d}")
    print(f"       Dog  {int(test_confusion_mat[1, 0]):3d}  {int(test_confusion_mat[1, 1]):3d}")

    print("\nPer-class metrics:")
    class_names = {0: "Cat", 1: "Dog"}
    for class_label in [0, 1]:
        metrics_dict = test_metrics[f'class_{class_label}']
        print(f"  {class_names[class_label]}:")
        print(f"    Precision: {metrics_dict['precision']:.5f}")
        print(f"    Recall: {metrics_dict['recall']:.5f}")
        print(f"    F1-score: {metrics_dict['f1']:.5f}")


if __name__ == '__main__':
    print(f"Using {DEVICE} device")
    # Enable auto-tuning of convolution algorithms
    torch.backends.cudnn.benchmark = True

    # Load data
    print("Loading Dogs vs Cats data...")
    train_loader, val_loader, test_loader = load_image_data()

    if DEBUG > 0:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

    # Create or load model
    if MODEL_STATE_FILE_NAME and os.path.exists(f"{MODEL_STATE_FILE_NAME}"):
        print(f"Loading pre-trained model: {MODEL_STATE_FILE_NAME}")
        model = load_model(f"{MODEL_STATE_FILE_NAME}")
    else:
        print("Creating new model...")
        model = DogsVsCatsCNN(CNN_CHANNELS, FC_HIDDEN_SIZES)

    model = model.to(DEVICE)

    # Train or test the model
    if not TEST_ONLY:
        print("Starting training...")
        # Create models directory
        os.makedirs("models", exist_ok=True)

        # Train the model
        train_model(model, train_loader, val_loader, MAX_EPOCHS)

    # Test model performance
    print("\nEvaluating model...")
    evaluate_model(model, test_loader)
