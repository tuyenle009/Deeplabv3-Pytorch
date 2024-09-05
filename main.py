import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex, Dice, MulticlassAccuracy

import segmentation_models_pytorch as smp
import shutil
from tqdm import tqdm

from src.dataset import SegDataset
from src.utils import img_transform, EarlyStopping

import os
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Train a deeplabv3 Plus model with specified hyperparameters.")

    # Adding arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data loading')
    parser.add_argument('--data_path', type=str, default='data/Semantic Segmentation.v3i.coco-segmentation', help='Path to the data directory')
    parser.add_argument('--model_save_path', type=str, default='trained_model', help='Path to save the trained model')
    parser.add_argument('--tensorboard_path', type=str, default='tensorboard', help='Path for TensorBoard logs')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for classification')

    # Parsing arguments
    args = parser.parse_args()
    return args

# Training fuction
def train(model, train_dataloader, device, optimizer, epoch, epochs, writer, criterion):
    all_losses = []
    model.train()
    train_progress = tqdm(train_dataloader, colour="cyan")
    for idx, img_mask in enumerate(train_progress):
        img = img_mask[0].float().to(device)  # img - B,C,H,W
        mask = img_mask[1].long().to(device)  # label - B,H,W
        y_pred = model(img)  # B, 4, H, W
        # Optimizer
        optimizer.zero_grad()
        loss = criterion(y_pred, mask)
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())

        # tracking the loss function
        writer.add_scalar("Train/Loss", np.mean(all_losses), epoch * len(train_dataloader) + idx)

        train_progress.set_description("TRAIN | Epoch: {}/{} | Iter: {}/{} | Loss: {:0.4f} | lr: {}".format(
            epoch + 1, epochs, idx + 1, len(train_dataloader), loss, optimizer.param_groups[0]['lr']))


# Evaluate function
def evaluate(model, val_dataloader, device, miou_metric, dice_metric, acc_metric):

    all_ious = []
    all_dices = []
    all_accs = []

    model.eval()
    with torch.no_grad():
        for idx, img_mask in enumerate(val_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].long().to(device)  # B W H

            y_pred = model(img)  # B, 4, H, W
            all_ious.append(miou_metric(y_pred, mask).cpu().item())
            all_dices.append(dice_metric(y_pred, mask).cpu().item())
            all_accs.append(acc_metric(y_pred, mask).cpu().item())

            if idx > 40: break

    miou = np.mean(all_ious)
    dice = np.mean(all_dices)
    acc = np.mean(all_accs)
    return acc, miou, dice


# Main function to run the training process
def main(args):
    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_path, exist_ok=True)

    # Remove and recreate the tensorboard directory
    if os.path.exists(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.makedirs(args.tensorboard_path)

    # Data augmentation and preprocessing for training and testing
    train_transform, test_transform = img_transform()  # Assume you have a function defined for this

    # Initialize datasets and dataloaders
    train_dataset = SegDataset(image_set="train", transform=train_transform)
    test_dataset = SegDataset(image_set="valid", transform=test_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                  drop_last=True)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                drop_last=True)

    # Set device for model and tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the device
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=args.num_classes
    ).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = smp.losses.JaccardLoss(mode="multiclass", log_loss=True)

    # Initialize performance meters for metrics calculation
    miou_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average='macro').to(device)
    dice_metric = Dice(num_classes=args.num_classes, average="macro").to(device)
    acc_metric = MulticlassAccuracy(num_classes=args.num_classes, average="macro").to(device)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=14, restore_best_weights=False)

    # Set up a learning rate scheduler to adjust the learning rate based on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4,
        threshold=1e-4, min_lr=0
    )

    # Track the best validation performance
    best_score = -1
    current_epoch = 0  # Allows for resuming training if necessary

    # Initialize TensorBoard writer
    writer = SummaryWriter(args.tensorboard_path)

    # Main training loop
    for epoch in range(current_epoch, args.epochs):

        # Training step
        train(model, train_dataloader, device, optimizer, epoch, args.epochs, writer, criterion)

        # Validation step
        accuracy, miou, dice = evaluate(
            model, val_dataloader, device,
            miou_metric, dice_metric, acc_metric
        )
        # Log metrics to TensorBoard
        writer.add_scalar("Valid/Accuracy", accuracy, epoch)
        writer.add_scalar("Valid/mIOU", miou, epoch)
        writer.add_scalar("Valid/DiceScore", dice, epoch)

        # Update learning rate based on validation metric (dice)
        scheduler.step(1 - miou)

        # Create checkpoint to save model state
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "dice_score": miou
        }

        # Save the most recent model checkpoint
        torch.save(checkpoint, os.path.join(args.model_save_path, "last.h5"))

        # Save the best model checkpoint based on Dice score
        if miou > best_score:
            torch.save(checkpoint, os.path.join(args.model_save_path, "best.h5"))
            best_score = miou

        # Check for early stopping criteria
        if early_stopping(model, 1 - miou):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Print validation results for monitoring
        print(
            f"VAL | Accuracy: {accuracy:.4f} | mIOU: {miou:.4f} | Dice Score: {dice:.4f} | EarlyStop: {early_stopping.status}")

if __name__ == '__main__':
    # Hyperparameters and paths
    args = get_args()
    main(args)