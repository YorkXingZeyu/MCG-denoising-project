import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam

def train_unet(unet, train_loader, val_loader, config):
    # Path configuration
    foldername = config["output_folder"]
    model_folder = os.path.join(foldername, "model_params")
    os.makedirs(model_folder, exist_ok=True)
    best_model_path = os.path.join(model_folder, "Unet1D_best.pth")
    final_model_path = os.path.join(model_folder, "Unet1D_final.pth")
    device = config["device"]

    unet.to(device)

    # Optimizer & Learning Rate Scheduler
    optimizer = Adam(
        unet.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]  # Add weight decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["train"]["step_size"], 
        gamma=config["train"]["gamma"]
    )

    best_loss = float("inf")
    counter = 0

    for epoch in range(config["train"]["epochs"]):
        unet.train()
        avg_loss = 0

        with tqdm(train_loader, desc=f"Training Unet Epoch {epoch}") as t:
            for clean_batch, noise_batch in t:
                clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                output = unet(noise_batch)

                # Compute total loss
                loss = F.mse_loss(output, clean_batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                t.set_postfix(avg_loss=avg_loss / len(train_loader))

        # Learning rate scheduling
        lr_scheduler.step()
        avg_epoch_loss = avg_loss / len(train_loader)
        avg_weighted_loss = avg_epoch_loss * config["train"]["unetloss_print_weight"]
        print(f"Epoch {epoch}: Avg Loss: {avg_weighted_loss:.4f}")

        # Validation
        val_loss = validate_unet(unet, val_loader, device, config)
        val_weighted_loss = val_loss * config["train"]["unetloss_print_weight"]
        print(f"Validation Loss: {val_weighted_loss:.4f}")

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(unet.state_dict(), best_model_path)
            best_weighted_loss = best_loss * config["train"]["unetloss_print_weight"]
            print(f"Best Validation Loss updated to {best_weighted_loss:.4f}. Model saved.")
        else:
            counter += 1
            if counter >= config["train"]["early_stopping"]:
                print("Early stopping triggered.")
                break

    # Save the final model
    torch.save(unet.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

def validate_unet(unet, val_loader, device, config):
    unet.eval()
    total_loss = 0

    with torch.no_grad():
        for clean_batch, noise_batch in val_loader:
            clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)

            output = unet(noise_batch)

            # Compute total loss
            loss = F.mse_loss(output, clean_batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)
