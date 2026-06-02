import matplotlib.pyplot as plt
import os

# pylint: disable=C0103
import numpy as np

import pylops
import pylops.optimization.sparsity

plt.close("all")

# Create output directory for figures
output_dir = "output/1D2_reconstructions"
os.makedirs(output_dir, exist_ok=True)
np.random.seed(10)
# Signal creation in frequency domain
ifreqs = [15, 7, 81]
amps = [1.0, 1.0, 1.0]
N = 200
nfft = 2**11
dt = 0.004
t = np.arange(N) * dt
f = np.fft.rfftfreq(nfft, dt)

FFTop = 10 * pylops.signalprocessing.FFT(N, nfft=nfft, real=True)

X = np.zeros(nfft // 2 + 1, dtype="complex128")
X[ifreqs] = amps
x = FFTop.H * X

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(f, np.abs(X), "k", lw=2)
axs[0].set_xlim(0, 30)
axs[0].set_title("Data(frequency domain)")
axs[1].plot(t, x, "k", lw=2)
axs[1].set_title("Data(time domain)")
axs[1].axis("tight")
plt.tight_layout()
plt.savefig(f"{output_dir}/01_original_data.png", dpi=150, bbox_inches="tight")
plt.show()

# subsampling locations
perc_subsampling = 0.20
Nsub = int(np.round(N * perc_subsampling))

iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])

# Create restriction operator
Rop = pylops.Restriction(N, iava, dtype="float64")

y = Rop * x
ymask = Rop.mask(x)

# Visualize data
fig = plt.figure(figsize=(12, 4))
plt.plot(t, x, "k", lw=3)
plt.plot(t, x, ".k", ms=20, label="all samples")
plt.plot(t, ymask, ".g", ms=15, label="available samples")
plt.legend()
plt.title("Data restriction")
plt.tight_layout()
plt.savefig(f"{output_dir}/02_data_restriction.png", dpi=150, bbox_inches="tight")
plt.show()

pista, niteri, costi = pylops.optimization.sparsity.ista(
    Rop * FFTop.H,
    y,
    niter=1000,
    eps=0.1,
    tol=1e-7,
)


xista = FFTop.H * pista

# FISTA reconstruction
pfista, niterfista, costfista = pylops.optimization.sparsity.fista(
    Rop * FFTop.H,
    y,
    niter=1000,
    eps=0.1,
    tol=1e-7,
)
xfista = FFTop.H * pfista

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(f, np.abs(X), "k", lw=3, label="Original")
axs[0].plot(f, np.abs(pista), "--r", lw=2, label="ISTA")
axs[0].plot(f, np.abs(pfista), "--b", lw=2, label="FISTA")
axs[0].set_xlim(0, 30)
axs[0].set_title("Frequency domain")
axs[0].legend()
axs[1].plot(t[iava], y, ".k", ms=20, label="Available samples")
axs[1].plot(t, x, "k", lw=3, label="Original")
axs[1].plot(t, xista, "--r", lw=2, label="ISTA")
axs[1].plot(t, xfista, "--b", lw=2, label="FISTA")
axs[1].set_title("Time domain")
axs[1].axis("tight")
axs[1].legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/03_ista_vs_fista_reconstruction.png", dpi=150, bbox_inches="tight")
plt.show()

# Convergence comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(costi, "r", lw=2, label="ISTA")
ax.semilogy(costfista, "b", lw=2, label="FISTA")
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title("Convergence Comparison: ISTA vs FISTA")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/04_convergence_comparison.png", dpi=150, bbox_inches="tight")
plt.show()