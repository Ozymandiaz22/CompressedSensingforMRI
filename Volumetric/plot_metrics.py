import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Enable pgf backend for TeX output
plt.rcParams['pgf.texsystem'] = 'xelatex'

# Set larger font sizes for A1 poster with Arial font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 12

metrics_file_1 = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 6 Graphs\\metrics_1.txt"
metrics_file_2 = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 6 Graphs\\metrics_2.txt"
graphs_output = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 6 Graphs"

def save_plot(fig, output_dir, filename):
    """Save plot as PNG, PGF, and wrapped in TeX."""
    # Sanitize filename to remove spaces
    filename = filename.replace(' ', '_')

    filepath_png = os.path.join(output_dir, f'{filename}.png')
    filepath_pgf = os.path.join(output_dir, f'{filename}.pgf')
    filepath_tex = os.path.join(output_dir, f'{filename}.tex')

    try:
        fig.savefig(filepath_png, dpi=100)
        print(f"  Saved: {filename}.png")
    except Exception as e:
        print(f"  Warning: Could not save PNG: {e}")
        return

    try:
        fig.savefig(filepath_pgf, format='pgf')
        print(f"  Saved: {filename}.pgf")

        # Wrap PGF in a standalone TeX file
        tex_content = f"""\\documentclass{{article}}
\\usepackage{{pgf}}
\\usepackage{{tikz}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}

\\pagestyle{{empty}}

\\begin{{document}}

\\input{{{filename}.pgf}}

\\end{{document}}
"""
        with open(filepath_tex, 'w') as f:
            f.write(tex_content)
        print(f"  Saved: {filename}.tex")

    except Exception as e:
        print(f"  Warning: Could not save PGF/TeX: {e}")

def parse_metrics_file(filepath):
    """Parse metrics file and return structured data."""
    data = defaultdict(lambda: defaultdict(dict))

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(', ')
            dataset = parts[0]
            undersampling_pct = float(parts[1])
            sampling_pattern = parts[2]

            metrics = {}
            for metric_str in parts[3:]:
                metric_name, metric_value = metric_str.split(': ')
                metrics[metric_name] = float(metric_value)

            data[dataset][sampling_pattern][undersampling_pct] = metrics

    return data

def plot_metrics_vs_undersampling(data, metric_name, output_dir):
    """Plot a specific metric vs undersampling percentage."""
    plt.figure(figsize=(10, 10))

    for dataset, patterns in data.items():
        for pattern, us_data in patterns.items():
            us_pcts = sorted(us_data.keys())
            metric_values = [us_data[pct][metric_name] for pct in us_pcts]
            plt.plot(us_pcts, metric_values, marker='o', label=f"{dataset} - {pattern}")

    plt.xlabel('Sampling (%)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Sampling Percentage')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'{metric_name}_vs_undersampling'
    save_plot(plt.gcf(), output_dir, filename)
    plt.close()

def plot_metrics_comparison(data, output_dir):
    """Plot all metrics in subplots."""
    metrics_to_plot = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]

        for dataset, patterns in data.items():
            for pattern, us_data in patterns.items():
                us_pcts = sorted(us_data.keys())
                metric_values = [us_data[pct][metric_name] for pct in us_pcts]
                ax.plot(us_pcts, metric_values, marker='o', label=f"{dataset} - {pattern}")

        ax.set_xlabel('Sampling Percentage (%)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Sampling %')
        ax.grid(True, alpha=0.3)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8)
    plt.tight_layout()

    filename = 'all_metrics_comparison'
    save_plot(fig, output_dir, filename)
    plt.close()

def plot_sampling_pattern_comparison(data, output_dir):
    """Compare different sampling patterns for each dataset."""
    for dataset, patterns in data.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        metrics_to_plot = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']

        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]

            for pattern, us_data in patterns.items():
                us_pcts = sorted(us_data.keys())
                metric_values = [us_data[pct][metric_name] for pct in us_pcts]
                ax.plot(us_pcts, metric_values, marker='s', label=pattern)

            ax.set_xlabel('Undersampling Percentage (%)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} - {dataset}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'{dataset}_sampling_patterns'
        save_plot(fig, output_dir, filename)
        plt.close()

def plot_average_metrics(data, output_dir):
    """Plot averaged metrics across all datasets."""
    metrics_to_plot = ['SSIM', 'PSNR', 'MSE', 'ROI SNR']

    # Collect averages per sampling pattern and undersampling percentage
    pattern_averages = defaultdict(lambda: defaultdict(list))

    for dataset, patterns in data.items():
        for pattern, us_data in patterns.items():
            for us_pct, metrics in us_data.items():
                for metric_name in metrics_to_plot:
                    pattern_averages[pattern][us_pct].append((metric_name, metrics[metric_name]))

    # Calculate averages for each metric, pattern, and US percentage
    pattern_avg_metrics = defaultdict(lambda: defaultdict(lambda: {}))
    for pattern, us_data in pattern_averages.items():
        for us_pct, metric_list in us_data.items():
            metric_dict = defaultdict(list)
            for metric_name, value in metric_list:
                metric_dict[metric_name].append(value)
            for metric_name in metrics_to_plot:
                if metric_dict[metric_name]:
                    pattern_avg_metrics[pattern][us_pct][metric_name] = np.mean(metric_dict[metric_name])

    # Plot each metric
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(10, 10))
        for pattern in sorted(pattern_avg_metrics.keys()):
            us_data = pattern_avg_metrics[pattern]
            us_pcts = sorted(us_data.keys())
            metric_values = [us_data[pct].get(metric_name, 0) for pct in us_pcts]
            plt.plot(us_pcts, metric_values, marker='o', linewidth=2, label=pattern)

        plt.xlabel('Sampling (%)')
        plt.ylabel(metric_name)
        plt.title(f'Average {metric_name} vs Undersampling Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f'average_{metric_name}_vs_undersampling'
        save_plot(plt.gcf(), output_dir, filename)
        plt.close()

    # Combined plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        for pattern in sorted(pattern_avg_metrics.keys()):
            us_data = pattern_avg_metrics[pattern]
            us_pcts = sorted(us_data.keys())
            metric_values = [us_data[pct].get(metric_name, 0) for pct in us_pcts]
            ax.plot(us_pcts, metric_values, marker='o', linewidth=2, label=pattern)

        ax.set_xlabel('Undersampling Percentage (%)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Average {metric_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = 'average_all_metrics_comparison'
    save_plot(fig, output_dir, filename)
    plt.close()

def create_metric_table(data, output_dir):
    """Create a summary table of metrics."""
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        for dataset, patterns in data.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"{'='*80}\n")

            for pattern, us_data in sorted(patterns.items()):
                f.write(f"\nSampling Pattern: {pattern}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{'US %':<10} {'SSIM':<12} {'PSNR':<12} {'MSE':<15} {'ROI SNR':<12}\n")
                f.write(f"{'-'*80}\n")

                for us_pct in sorted(us_data.keys()):
                    metrics = us_data[us_pct]
                    f.write(f"{us_pct:<10.1f} {metrics['SSIM']:<12.4f} {metrics['PSNR']:<12.4f} {metrics['MSE']:<15.6f} {metrics['ROI SNR']:<12.4f}\n")

# Main execution
if __name__ == "__main__":
    print("Reading metrics from files...")
    data_1 = parse_metrics_file(metrics_file_1)

    print(f"Found data for datasets: {list(data_1.keys())}\n")

    print("Creating plots...")
    plot_metrics_vs_undersampling(data_1, 'SSIM', graphs_output)
    plot_metrics_vs_undersampling(data_1, 'PSNR', graphs_output)
    plot_metrics_vs_undersampling(data_1, 'MSE', graphs_output)
    plot_metrics_vs_undersampling(data_1, 'ROI SNR', graphs_output)

    print("Creating comparison plots...")
    plot_metrics_comparison(data_1, graphs_output)
    plot_sampling_pattern_comparison(data_1, graphs_output)

    print("Creating average plots across all datasets...")
    plot_average_metrics(data_1, graphs_output)

    print("Creating summary table...")
    create_metric_table(data_1, graphs_output)

    print(f"\nAll plots saved to {graphs_output}")
