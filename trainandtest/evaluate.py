import os
import torch
import pandas as pd
from utils.metrics import SSD, MAD, COS_SIM, SNR

def evaluate_fold(unet, data_loader, config, fold=None):

    device = config["device"]
    out_folder = config["output_folder"]
    model_folder = os.path.join(out_folder, "model_params")

    if fold is not None:
        weight_file = os.path.join(model_folder, f"Unet1D_fold{fold}.pth")
    else:
        weight_file = os.path.join(model_folder, "Unet1D_best.pth")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"权重文件不存在: {weight_file}")
    unet.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()
    print(f"Loaded weights from {weight_file}")

    ssd = mad = prd = cos_sim = 0.0
    snr_in = snr_out = snr_imp = 0.0
    count = 0

    with torch.no_grad():
        for clean, noisy in data_loader:
            clean, noisy = clean.to(device), noisy.to(device)
            output = unet(noisy)

            # permute to [B, L, C]
            c_np = clean.permute(0, 2, 1).cpu().numpy()
            n_np = noisy.permute(0, 2, 1).cpu().numpy()
            o_np = output.permute(0, 2, 1).cpu().numpy()

            B = o_np.shape[0]
            count += B

            ssd       += SSD(c_np, o_np).sum()
            mad       += MAD(c_np, o_np).sum()
            prd       += PRD(c_np, o_np).sum()
            cos_sim   += COS_SIM(c_np, o_np).sum()
            snr_in    += SNR(c_np, n_np).sum()
            snr_out   += SNR(c_np, o_np).sum()
            snr_imp   += SNR_improvement(n_np, o_np, c_np).sum()

    metrics = {
        "SSD":             ssd   / count,
        "MAD":             mad   / count,
        "PRD":             prd   / count,
        "CosineSim":       cos_sim / count,
        "SNR_Input":       snr_in   / count,
        "SNR_Output":      snr_out  / count,
        "SNR_Improvement": snr_imp  / count,
    }

    df = pd.DataFrame({k: [v] for k, v in metrics.items()})
    fname = f"evaluation_results_fold{fold}.xlsx" if fold is not None else "evaluation_results.xlsx"
    out_path = os.path.join(out_folder, fname)
    df.to_excel(out_path, index=False)
    print(f"Saved fold {fold or 'best'} results to {out_path}")

    return metrics
