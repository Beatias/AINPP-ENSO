import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def save_epoch_sample(model, loader, epoch, device, save_dir="samples"):
    """
    Generates and saves a comparative plot (Obs vs Pred) for the first validation sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. Get a single validation batch
    batch = next(iter(loader))
    inputs, targets = batch[0], batch[1]

    # 2. Prepare data exactly as in training.
    # ENSO channels mode: [B, Tin, 4, H, W]
    # Standard dataset:    [B, Tin, 1, H, W]
    inputs = inputs.to(device)
    targets = targets.to(device)
    print(
        f"Max target value (mm/h): {targets.max().item():.2f}, "
        f"Shape: {targets.shape}"
    )

    with torch.no_grad():
        outputs_log = model(inputs)

        # 3. Revert to mm/h for visualization (expm1)
        preds_mm = outputs_log.cpu().numpy()
        targets_mm = targets[:, :, 0].cpu().numpy()

    # 4. Configurar o Plot
    batch_idx = 0  # Pega a primeira amostra do batch
    timesteps = targets_mm.shape[1]  # Quantos tempos futuros (M_OUT)

    # Cria figura: 2 linhas (Obs, Pred) x N colunas (Tempos)
    fig, axes = plt.subplots(2, timesteps, figsize=(4 * timesteps, 6))

    # Treatment if there is only 1 timestep (for axes to behave like a matrix)
    if timesteps == 1:
        axes = axes.reshape(2, 1)

    # Define common color scale (based on maximum observation for fair comparison)
    # vmax = targets_mm[batch_idx].max() + 1.0

    for t in range(timesteps):
        # Top Row: Observation (Ground Truth)
        ax_obs = axes[0, t]
        ax_obs.imshow(targets_mm[batch_idx, t], cmap="jet", origin="upper")
        ax_obs.set_title(f"Obs T+{t+1}")
        ax_obs.axis("off")

        # Bottom Row: Prediction
        ax_pred = axes[1, t]
        ax_pred.imshow(preds_mm[batch_idx, t, 0], cmap="jet", origin="upper")
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch+1}_sample.png")
    plt.close(fig)  # Close to free RAM memory
