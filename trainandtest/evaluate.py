import torch
import shutil
from utils.metrics import SSD, MAD, PRD, COS_SIM, SNR, SNR_improvement
from utils.plot import plot_signals
import os
import pandas as pd

def evaluate_model(unet, test_loader, config):
    device = config["device"]

    # Load model weights
    foldername = config["output_folder"]
    model_folder = os.path.join(foldername, "model_params")
    unet_model_path = os.path.join(model_folder, 'Unet1D_best.pth')

    if not os.path.exists(unet_model_path):
        raise FileNotFoundError("Model weights not found. Ensure Unet weights are available.")

    unet.load_state_dict(torch.load(unet_model_path, map_location=device, weights_only=True))
    unet.to(device)
    print(f"Loaded Unet weights from {unet_model_path}")

    unet.eval()

    # Initialize evaluation metrics
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0

    all_clean_signals = []
    all_output_signals = []

    with torch.no_grad():
        for batch_no, (clean_batch, noise_batch) in enumerate(test_loader):
            clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)

            output = unet(noise_batch)

            plot_signals(clean_batch, noise_batch, output, batch_no, foldername=foldername, filename="images_results")

            # Permute for evaluation
            clean_batch = clean_batch.permute(0, 2, 1)
            noise_batch = noise_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1)

            clean_numpy = clean_batch.cpu().numpy()
            noisy_numpy = noise_batch.cpu().numpy()
            output_numpy = output.cpu().numpy()

            eval_points += len(output)

            # Compute metrics
            ssd_total += SSD(clean_numpy, output_numpy).sum()
            mad_total += MAD(clean_numpy, output_numpy).sum()
            prd_total += PRD(clean_numpy, output_numpy).sum()
            cos_sim_total += COS_SIM(clean_numpy, output_numpy).sum()
            snr_noise += SNR(clean_numpy, noisy_numpy).sum()
            snr_recon += SNR(clean_numpy, output_numpy).sum()
            snr_improvement += SNR_improvement(noisy_numpy, output_numpy, clean_numpy).sum()

            # Store signals for visualization
            all_clean_signals.extend(clean_numpy)
            all_output_signals.extend(output_numpy)

    # Compute average evaluation metrics
    avg_ssd = ssd_total / eval_points
    avg_mad = mad_total / eval_points
    avg_prd = prd_total / eval_points
    avg_cos_sim = cos_sim_total / eval_points
    avg_snr_noise = snr_noise / eval_points
    avg_snr_recon = snr_recon / eval_points
    avg_snr_improvement = snr_improvement / eval_points

    # Save evaluation results
    results = {
        "SSD": [avg_ssd],
        "MAD": [avg_mad],
        "PRD": [avg_prd],
        "Cosine Similarity": [avg_cos_sim],
        "SNR Input": [avg_snr_noise],
        "SNR Output": [avg_snr_recon],
        "SNR Improvement": [avg_snr_improvement],
    }
    df = pd.DataFrame(results)
    file_path = os.path.join(foldername, 'evaluation_results.xlsx')
    df.to_excel(file_path, index=False)
    print(f"Evaluation results saved to {file_path}")

    print("\nEvaluation Results:")
    print(f"  SSD: {avg_ssd}")
    print(f"  MAD: {avg_mad}")
    print(f"  PRD: {avg_prd}")
    print(f"  Cosine Similarity: {avg_cos_sim}")
    print(f"  SNR Input: {avg_snr_noise}")
    print(f"  SNR Output: {avg_snr_recon}")
    print(f"  SNR Improvement: {avg_snr_improvement}")

    # Compress the output folder
    archive_path = shutil.make_archive(foldername, 'zip', foldername)
    print(f"Folder '{foldername}' compressed to '{archive_path}'")
