import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from collections import defaultdict
import methods.sphereinisocentre as spiic
import skimage.metrics as metrics

# Paths
nifti_path = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 8 Nifti Outputs 2"
groundtruth_path = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 6 Groundtruth Results"
output_path = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 8 Graphs"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load all nifti files
nifti_files = [f for f in os.listdir(nifti_path) if f.endswith('.nii')]
print(f"Found {len(nifti_files)} nifti files")

# Load ground truth files
groundtruth_files = {}
for f in os.listdir(groundtruth_path):
    if f.endswith('.nii'):
        filepath = os.path.join(groundtruth_path, f)
        img = nib.load(filepath)
        groundtruth_files[f] = img.get_fdata()
print(f"Found {len(groundtruth_files)} ground truth files")

# Parse filenames and organize by wavelet and dataset
def parse_filename(filename):
    # Format: image2_5684_000_0_coif1_Reg=5.0.nii
    parts = filename.replace('.nii', '').split('_')
    dataset_id = parts[1]
    # Wavelet is now before _Reg=
    wavelet = parts[-2].split('Reg=')[0].rstrip('_')
    return dataset_id, wavelet

# Organize data by (dataset, wavelet)
recon_data = defaultdict(dict)
for f in nifti_files:
    dataset_id, wavelet = parse_filename(f)
    filepath = os.path.join(nifti_path, f)
    img = nib.load(filepath)
    recon_data[(dataset_id, wavelet)] = img.get_fdata()

print(f"Organized {len(recon_data)} reconstructions")

# Compute metrics
def compute_metrics(recon, ground_truth):
    ssim = metrics.structural_similarity(ground_truth, recon, data_range=ground_truth.max() - ground_truth.min())
    psnr = metrics.peak_signal_noise_ratio(ground_truth, recon, data_range=ground_truth.max() - ground_truth.min())
    mse = metrics.mean_squared_error(ground_truth, recon)
    roi_mask = spiic.sphere_in_isocentre(ground_truth, percentradius=0.5)
    roi_signal = np.mean(recon[roi_mask == 1])
    noise = np.std(recon[roi_mask == 0])
    roi_snr = 10 * np.log10(roi_signal / noise) if noise > 0 else float('inf')
    return ssim, psnr, mse, roi_snr

# Compute metrics for each reconstruction
metrics_data = {}
for (dataset_id, wavelet), recon in recon_data.items():
    gt_filename = f"images_{dataset_id}.nii"
    if gt_filename in groundtruth_files:
        gt = groundtruth_files[gt_filename]
        ssim, psnr, mse, roi_snr = compute_metrics(recon, gt)
        metrics_data[(dataset_id, wavelet)] = {
            'SSIM': ssim,
            'PSNR': psnr,
            'MSE': mse,
            'ROI SNR': roi_snr
        }
        print(f"Dataset {dataset_id}, Wavelet {wavelet}: SSIM={ssim:.4f}, PSNR={psnr:.4f}, MSE={mse:.6f}, ROI SNR={roi_snr:.4f}")

# Save metrics to file
metrics_file = os.path.join(output_path, "wavelet_metrics.txt")
with open(metrics_file, 'w') as f:
    for (dataset_id, wavelet), metric_vals in sorted(metrics_data.items()):
        f.write(f"Dataset: {dataset_id}, Wavelet: {wavelet}, SSIM: {metric_vals['SSIM']}, PSNR: {metric_vals['PSNR']}, MSE: {metric_vals['MSE']}, ROI SNR: {metric_vals['ROI SNR']}\n")
print(f"\nMetrics saved to {metrics_file}")

# Organize metrics by dataset and wavelet
metrics_by_dataset = defaultdict(dict)
metrics_by_wavelet = defaultdict(dict)
all_datasets = set()
all_wavelets = set()

for (dataset_id, wavelet), vals in metrics_data.items():
    metrics_by_dataset[dataset_id][wavelet] = vals
    metrics_by_wavelet[wavelet][dataset_id] = vals
    all_datasets.add(dataset_id)
    all_wavelets.add(wavelet)

all_datasets = sorted(all_datasets)
all_wavelets = sorted(all_wavelets)

print(f"\nDatasets: {all_datasets}")
print(f"Wavelets: {all_wavelets}")

# Plot metrics for each dataset comparing wavelets
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8

for dataset_id in all_datasets:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Metrics for Dataset {dataset_id} by Wavelet')
    axes = axes.flatten()

    metrics_names = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        wavelets = []
        values = []

        for wavelet in all_wavelets:
            if wavelet in metrics_by_dataset[dataset_id]:
                wavelets.append(wavelet)
                values.append(metrics_by_dataset[dataset_id][wavelet][metric_name])

        ax.bar(wavelets, values, color='steelblue', alpha=0.7)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Wavelet')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    filename = os.path.join(output_path, f'dataset_{dataset_id}_wavelet_comparison.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Plot metrics for each wavelet comparing datasets
for wavelet in all_wavelets:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Metrics for Wavelet {wavelet} by Dataset')
    axes = axes.flatten()

    metrics_names = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        datasets = []
        values = []

        for dataset_id in all_datasets:
            if dataset_id in metrics_by_wavelet[wavelet]:
                datasets.append(dataset_id)
                values.append(metrics_by_wavelet[wavelet][dataset_id][metric_name])

        ax.bar(datasets, values, color='coral', alpha=0.7)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Dataset')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    filename = os.path.join(output_path, f'wavelet_{wavelet}_dataset_comparison.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create combined comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wavelet Metrics Comparison (All Datasets)')
axes = axes.flatten()

metrics_names = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']
colors = plt.cm.Set3(np.linspace(0, 1, len(all_wavelets)))

for idx, metric_name in enumerate(metrics_names):
    ax = axes[idx]

    x = np.arange(len(all_datasets))
    width = 0.12

    for wavelet_idx, wavelet in enumerate(all_wavelets):
        values = []
        for dataset_id in all_datasets:
            if dataset_id in metrics_by_wavelet[wavelet]:
                values.append(metrics_by_wavelet[wavelet][dataset_id][metric_name])
            else:
                values.append(0)

        offset = width * (wavelet_idx - len(all_wavelets)/2 + 0.5)
        ax.bar(x + offset, values, width, label=wavelet, color=colors[wavelet_idx], alpha=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Dataset and Wavelet')
    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
filename = os.path.join(output_path, 'all_wavelets_comparison.png')
plt.savefig(filename, dpi=100, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Create summary table
summary_file = os.path.join(output_path, "wavelet_metrics_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*100 + "\n")
    f.write("Wavelet Metrics Summary\n")
    f.write("="*100 + "\n\n")

    for dataset_id in all_datasets:
        f.write(f"\nDataset: {dataset_id}\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Wavelet':<12} {'SSIM':<12} {'PSNR':<12} {'MSE':<15} {'ROI SNR':<12}\n")
        f.write("-"*100 + "\n")

        for wavelet in all_wavelets:
            if wavelet in metrics_by_dataset[dataset_id]:
                vals = metrics_by_dataset[dataset_id][wavelet]
                f.write(f"{wavelet:<12} {vals['SSIM']:<12.4f} {vals['PSNR']:<12.4f} {vals['MSE']:<15.6f} {vals['ROI SNR']:<12.4f}\n")

print(f"\nSummary saved to {summary_file}")
print(f"\nAll plots and metrics saved to {output_path}")
