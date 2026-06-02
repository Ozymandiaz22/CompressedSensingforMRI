import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import json
import gc
import methods.sphereinisocentre as spiic
import skimage.metrics as metrics

nifti_path_1 = "C:\\Users\\osman\\Documents\\Final_FYP_Dataset\\Final_1_nifti"
nifti_path_2 = "C:\\Users\\osman\\Documents\\Final_FYP_Dataset\\Final_2_nifti"
truth_nifti_path = "C:\\Users\\osman\\Documents\\Final_FYP_Dataset\\Final_1 Groundtruth"
truth_2_nifti_path = "C:\\Users\\osman\\Documents\\Final_FYP_Dataset\\Final_2 Groundtruth"
graphs = "C:\\Users\\osman\\Documents\\Final_FYP_Dataset\\Analysis_Graphs"
os.makedirs(graphs, exist_ok=True)

BATCH_SIZE = 50
CHECKPOINT_FILE = os.path.join(graphs, "checkpoint.json")
RESULTS_FILE = os.path.join(graphs, "metrics_intermediate.json")

##split the filename to extract: acquisition, image_id, dataset, under-sampling percentage, and sampling pattern
def parse_filename(filename):
    # Format: 1_image_5684_000_0_haar_Reg=5.0_Data#1_SamplingMask=50_gauss.nii
    parts = filename.replace('.nii', '').split('_')
    acquisition = parts[0]
    image_id = parts[2]
    acquisition_image = f"{acquisition}_{image_id}"

    dataset = None
    for part in parts:
        if part.startswith('Data#'):
            dataset = part
            break

    undersampling_percentage = None
    for part in parts:
        if part.startswith('SamplingMask='):
            undersampling_percentage = part.split('=')[1]
            break

    sampling_pattern = parts[-1]
    return acquisition_image, dataset, undersampling_percentage, sampling_pattern

def compute_metrics(recon, ground_truth):
    ssim = metrics.structural_similarity(ground_truth, recon, data_range=ground_truth.max() - ground_truth.min())
    psnr = metrics.peak_signal_noise_ratio(ground_truth, recon, data_range=ground_truth.max() - ground_truth.min())
    mse = metrics.mean_squared_error(ground_truth, recon)
    roi_mask = spiic.sphere_in_isocentre(ground_truth, percentradius=0.5)
    roi_signal = np.mean(recon[roi_mask == 1])
    noise = np.std(recon[roi_mask == 0])
    roi_snr = 10 * np.log10(roi_signal / noise) if noise > 0 else float('inf')
    return ssim, psnr, mse, roi_snr

##load truth files from subdirectories organized by image ID
print("Loading ground truth files...")
truth_data = {}
if os.path.exists(truth_nifti_path):
    for folder in os.listdir(truth_nifti_path):
        folder_path = os.path.join(truth_nifti_path, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.nii'):
                    full_path = os.path.join(folder_path, f)
                    truth_img = nib.load(full_path)
                    truth_data[f] = truth_img.get_fdata()

print(f"Loaded {len(truth_data)} ground truth files from path 1")

truth_2_data = {}
if os.path.exists(truth_2_nifti_path):
    for folder in os.listdir(truth_2_nifti_path):
        folder_path = os.path.join(truth_2_nifti_path, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.nii'):
                    full_path = os.path.join(folder_path, f)
                    truth_img = nib.load(full_path)
                    truth_2_data[f] = truth_img.get_fdata()

print(f"Loaded {len(truth_2_data)} ground truth files from path 2")

# Get list of reconstruction files
nifti_files_1 = [f for f in os.listdir(nifti_path_1) if f.endswith('.nii')]
nifti_files_2 = [f for f in os.listdir(nifti_path_2) if f.endswith('.nii')] if os.path.exists(nifti_path_2) else []

print(f"Found {len(nifti_files_1)} files in path 1")
print(f"Found {len(nifti_files_2)} files in path 2")

def load_checkpoint():
    """Load checkpoint to resume processing"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"path1_batch": 0, "path2_batch": 0}
    return {"path1_batch": 0, "path2_batch": 0}

def save_checkpoint(checkpoint):
    """Save checkpoint for resuming"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def load_results():
    """Load intermediate results"""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                ssim_vals = {}
                psnr_vals = {}
                mse_vals = {}
                snr_vals = {}
                for k, v in data.items():
                    key = tuple(k.split('|')[:-1])
                    metric_type = k.split('|')[-1]
                    if metric_type == 'ssim':
                        ssim_vals[key] = v
                    elif metric_type == 'psnr':
                        psnr_vals[key] = v
                    elif metric_type == 'mse':
                        mse_vals[key] = v
                    elif metric_type == 'snr':
                        snr_vals[key] = v
                return ssim_vals, psnr_vals, mse_vals, snr_vals
        except:
            return {}, {}, {}, {}
    return {}, {}, {}, {}

def save_results(ssim_dict, psnr_dict, mse_dict, snr_dict):
    """Save intermediate results"""
    combined = {}
    for k, v in ssim_dict.items():
        key_str = '|'.join(k) + '|ssim'
        combined[key_str] = v
    for k, v in psnr_dict.items():
        key_str = '|'.join(k) + '|psnr'
        combined[key_str] = v
    for k, v in mse_dict.items():
        key_str = '|'.join(k) + '|mse'
        combined[key_str] = v
    for k, v in snr_dict.items():
        key_str = '|'.join(k) + '|snr'
        combined[key_str] = v
    with open(RESULTS_FILE, 'w') as f:
        json.dump(combined, f, indent=2)

def process_batch(file_list, nifti_path, truth_data, batch_num, total_batches):
    """Process a batch of NIFTI files and return metrics"""
    batch_metrics = {}
    for filename in file_list:
        acq_image, dataset, undersamp, pattern = parse_filename(filename)
        acq_num, image_id = acq_image.split('_')
        ground_truth_filename = f"images_{image_id}_{acq_num}.nii"

        if ground_truth_filename in truth_data:
            try:
                recon_path = os.path.join(nifti_path, filename)
                recon_img = nib.load(recon_path)
                recon = recon_img.get_fdata()

                ground_truth = truth_data[ground_truth_filename]
                ssim, psnr, mse, roi_snr = compute_metrics(recon, ground_truth)

                key = (acq_image, dataset, undersamp, pattern)
                batch_metrics[key] = {'ssim': ssim, 'psnr': psnr, 'mse': mse, 'roi_snr': roi_snr}
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    gc.collect()
    return batch_metrics

checkpoint = load_checkpoint()
ssim_values_1, psnr_values_1, mse_values_1, roi_snr_values_1 = load_results()

# Process path 1 in batches
print("\nProcessing path 1 reconstructions...")
print(f"Resuming from batch {checkpoint['path1_batch']}")
total_batches_1 = (len(nifti_files_1) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(checkpoint['path1_batch'], total_batches_1):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(nifti_files_1))
    batch_files = nifti_files_1[start_idx:end_idx]

    print(f"  Batch {batch_idx + 1}/{total_batches_1} ({start_idx + 1}-{end_idx}/{len(nifti_files_1)})")
    batch_metrics = process_batch(batch_files, nifti_path_1, truth_data, batch_idx + 1, total_batches_1)

    for key, metrics_dict in batch_metrics.items():
        ssim_values_1[key] = metrics_dict['ssim']
        psnr_values_1[key] = metrics_dict['psnr']
        mse_values_1[key] = metrics_dict['mse']
        roi_snr_values_1[key] = metrics_dict['roi_snr']

    checkpoint['path1_batch'] = batch_idx + 1
    save_checkpoint(checkpoint)
    save_results(ssim_values_1, psnr_values_1, mse_values_1, roi_snr_values_1)
    print(f"    Processed {len(batch_metrics)} files, total: {len(ssim_values_1)}")

# Process path 2 in batches if it exists
if nifti_files_2:
    print("\nProcessing path 2 reconstructions...")
    print(f"Resuming from batch {checkpoint['path2_batch']}")
    total_batches_2 = (len(nifti_files_2) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(checkpoint['path2_batch'], total_batches_2):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(nifti_files_2))
        batch_files = nifti_files_2[start_idx:end_idx]

        print(f"  Batch {batch_idx + 1}/{total_batches_2} ({start_idx + 1}-{end_idx}/{len(nifti_files_2)})")
        batch_metrics = process_batch(batch_files, nifti_path_2, truth_2_data, batch_idx + 1, total_batches_2)

        for key, metrics_dict in batch_metrics.items():
            ssim_values_1[key] = metrics_dict['ssim']
            psnr_values_1[key] = metrics_dict['psnr']
            mse_values_1[key] = metrics_dict['mse']
            roi_snr_values_1[key] = metrics_dict['roi_snr']

        checkpoint['path2_batch'] = batch_idx + 1
        save_checkpoint(checkpoint)
        save_results(ssim_values_1, psnr_values_1, mse_values_1, roi_snr_values_1)
        print(f"    Processed {len(batch_metrics)} files, total: {len(ssim_values_1)}")

print(f"\nComputed metrics for {len(ssim_values_1)} reconstructions")

os.remove(CHECKPOINT_FILE) if os.path.exists(CHECKPOINT_FILE) else None
os.remove(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else None

##save the metrics in a text file
print("Saving metrics to file...")
with open(os.path.join(graphs, "metrics.txt"), 'w') as f:
    f.write("Acquisition,Image_ID,Dataset,Undersampling_Percentage,Sampling_Pattern,SSIM,PSNR,MSE,ROI_SNR\n")
    for key in sorted(ssim_values_1.keys()):
        acquisition_image, dataset, undersampling_percentage, sampling_pattern = key
        f.write(f"{acquisition_image},{dataset},{undersampling_percentage},{sampling_pattern},{ssim_values_1[key]:.4f},{psnr_values_1[key]:.4f},{mse_values_1[key]:.4f},{roi_snr_values_1[key]:.4f}\n")

# Create organized metric visualizations
print("Generating graphs...")
unique_datasets = set(k[1] for k in ssim_values_1.keys())
unique_patterns = set(k[3] for k in ssim_values_1.keys())
undersampling_values = sorted(set(k[2] for k in ssim_values_1.keys()), key=lambda x: int(x) if x.isdigit() else 0)

# Plot 1: SSIM vs Undersampling for each sampling pattern
plt.figure(figsize=(12, 6))
for pattern in sorted(unique_patterns):
    data_points = [(int(k[2]) if k[2].isdigit() else 0, v) for k, v in ssim_values_1.items() if k[3] == pattern]
    if data_points:
        data_points.sort()
        x_vals, y_vals = zip(*data_points)
        plt.plot(x_vals, y_vals, marker='o', label=pattern, linewidth=2, markersize=6)
plt.xlabel('Undersampling Percentage (%)', fontsize=11)
plt.ylabel('SSIM', fontsize=11)
plt.title('SSIM vs Undersampling Rate by Sampling Pattern', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graphs, 'ssim_vs_undersampling.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: PSNR vs Undersampling for each sampling pattern
plt.figure(figsize=(12, 6))
for pattern in sorted(unique_patterns):
    data_points = [(int(k[2]) if k[2].isdigit() else 0, v) for k, v in psnr_values_1.items() if k[3] == pattern]
    if data_points:
        data_points.sort()
        x_vals, y_vals = zip(*data_points)
        plt.plot(x_vals, y_vals, marker='s', label=pattern, linewidth=2, markersize=6)
plt.xlabel('Undersampling Percentage (%)', fontsize=11)
plt.ylabel('PSNR (dB)', fontsize=11)
plt.title('PSNR vs Undersampling Rate by Sampling Pattern', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graphs, 'psnr_vs_undersampling.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: ROI SNR vs Undersampling for each sampling pattern
plt.figure(figsize=(12, 6))
for pattern in sorted(unique_patterns):
    data_points = [(int(k[2]) if k[2].isdigit() else 0, v) for k, v in roi_snr_values_1.items() if k[3] == pattern and v != float('inf')]
    if data_points:
        data_points.sort()
        x_vals, y_vals = zip(*data_points)
        plt.plot(x_vals, y_vals, marker='^', label=pattern, linewidth=2, markersize=6)
plt.xlabel('Undersampling Percentage (%)', fontsize=11)
plt.ylabel('ROI SNR (dB)', fontsize=11)
plt.title('ROI SNR vs Undersampling Rate by Sampling Pattern', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graphs, 'roi_snr_vs_undersampling.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Metric comparison bar chart for best configuration
if ssim_values_1:
    best_key = max(ssim_values_1.keys(), key=lambda x: ssim_values_1[x])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics_names = ['SSIM', 'PSNR', 'ROI SNR']
    best_values = [ssim_values_1[best_key], psnr_values_1[best_key], roi_snr_values_1[best_key]]

    for i, (ax, metric, value) in enumerate(zip(axes, metrics_names, best_values)):
        ax.bar(['Best Config'], [value], color=['#2ecc71'], width=0.4)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.text(0, value, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    acquisition_image, dataset, undersampling, pattern = best_key
    fig.suptitle(f'Best Configuration: {acquisition_image}, {dataset}, {undersampling}%, {pattern}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs, 'best_configuration_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Graphs saved to {graphs}")
print("Analysis complete!")
