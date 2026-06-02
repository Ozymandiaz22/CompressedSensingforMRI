import pywt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path('output/wavelets')
output_dir.mkdir(parents=True, exist_ok=True)

# List of wavelets to plot
wavelets = ['haar', 'db4', 'db6', 'sym4', 'sym6', 'coif1', 'coif3', 'coif5']

for wavelet_name in wavelets:
    try:
        # Get wavelet
        wavelet = pywt.Wavelet(wavelet_name)

        # Get the wavelet function data
        result = wavelet.wavefun(level=8)

        # Handle both orthogonal and biorthogonal wavelets
        if wavelet.orthogonal:
            phi, psi, x = result
        else:
            phi_d, psi_d, phi_r, psi_r, x = result
            psi = psi_d  # Use decomposition wavelet for visualization

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, psi, linewidth=2, color='steelblue')
        ax.set_title(f'Wavelet: {wavelet_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Save figure
        output_path = output_dir / f'{wavelet_name}_wavelet.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

        plt.close()

    except Exception as e:
        print(f'Error processing {wavelet_name}: {e}')

print(f'\nAll wavelet images saved to {output_dir}')
