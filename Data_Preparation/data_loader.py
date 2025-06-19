import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from Data_Preparation.data_preparation import Data_Prepar_1, Data_Prepar_2

def prepare_data(config):
    folder = config["output_folder"]
    if folder == "MCG_real_result":
        X_np, y_np = Data_Prepar_1()
    elif folder == "MCG_simulated_result":
        X_np, y_np = Data_Prepar_2()
    else:
        raise ValueError(f"Unknown output_folder: {folder}")

    # to Torch tensors
    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np)

    # permute → (B, C, L)
    X = X.permute(0, 2, 1)
    y = y.permute(0, 2, 1)

    # pack into one dataset (we keep label first to match your previous order)
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

    # 2) of the 80% remaining, allocate 12.5% to validation
    #    (0.125 * 0.80 = 0.10 of total)
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
