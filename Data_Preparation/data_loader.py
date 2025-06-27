import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from Data_Preparation.data_preparation import Data_Prepar_1, Data_Prepar_2

def prepare_full_dataset(config):

    folder = config["output_folder"]
    if folder == "MCG_real_result":
        X_np, y_np = Data_Prepar_1()
    elif folder == "MCG_simulated_result":
        X_np, y_np = Data_Prepar_2()
    else:
        raise ValueError(f"Unknown output_folder: {folder}")

    X = torch.FloatTensor(X_np).permute(0, 2, 1)
    y = torch.FloatTensor(y_np).permute(0, 2, 1)

    return TensorDataset(y, X)

def prepare_data(config):

    folder = config["output_folder"]
    if folder == "MCG_real_result":
        X_np, y_np = Data_Prepar_1()
    elif folder == "MCG_simulated_result":
        X_np, y_np = Data_Prepar_2()
    else:
        raise ValueError(f"Unknown output_folder: {folder}")

    # to Torch tensors + permute → (B, C, L)
    X = torch.FloatTensor(X_np).permute(0, 2, 1)
    y = torch.FloatTensor(y_np).permute(0, 2, 1)

    # pack into one dataset (label first)
    full_ds = TensorDataset(y, X)
    N = len(full_ds)
    indices = list(range(N))

    # 1) split off 20% for test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # 2) 70% for train and 10% for val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125, 
        random_state=42,
        shuffle=True
    )

    # build subsets
    train_set = Subset(full_ds, train_idx)
    val_set   = Subset(full_ds, val_idx)
    test_set  = Subset(full_ds, test_idx)

    # data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config["test"]["batch_size"],
        shuffle=False
    )

    print(f"→ Samples — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader
