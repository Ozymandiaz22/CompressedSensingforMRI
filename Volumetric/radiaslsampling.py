# 3D MPRAGE Random Undersampled K-space Mask Generator
#
# Adapted from original by Gustav Strijkers
# g.j.strijkers@amsterdamumc.nl
#
# Refactored to accept a 3D k-space numpy array and return a 3D volumetric sampling mask.

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_sample_unique(N: int, weights: np.ndarray, k: int) -> np.ndarray:
    """
    Draw k unique indices from [0, N) without replacement using weighted sampling.

    Parameters
    ----------
    N       : Total number of candidates.
    weights : Non-negative weight for each candidate (need not sum to 1).
    k       : Number of unique samples to draw.

    Returns
    -------
    indices : 1-D int array of length k.
    """
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    indices = []
    available = np.ones(N, dtype=bool)

    for _ in range(k):
        w_curr = weights.copy()
        w_curr[~available] = 0.0
        w_curr /= w_curr.sum()
        cdf = np.cumsum(w_curr)
        r = np.random.rand()
        sel = np.searchsorted(cdf, r, side='right')
        indices.append(sel)
        available[sel] = False

    return np.array(indices)


def _poisson_pattern(
    size_y: int,
    size_z: int,
    variable_density: float,
    accel_factor: float,
    elliptical: bool,
    rand_seed: int,
    calib_reg_y: int,
    calib_reg_z: int,
    shot_length: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Generate a 2D Poisson-disc–inspired variable-density random sampling pattern
    for the (ky, kz) phase-encode plane.

    Parameters
    ----------
    size_y          : Number of k-space points along ky.
    size_z          : Number of k-space points along kz.
    variable_density: Density exponent (0 = uniform, >0 = centre-heavy).
    accel_factor    : Target acceleration factor (R ≥ 1).
    elliptical      : If True, apply an elliptical shutter to the PE plane.
    rand_seed       : Random seed for reproducibility.
    calib_reg_y     : Fully-sampled calibration region size along ky.
    calib_reg_z     : Fully-sampled calibration region size along kz.
    shot_length     : Number of phase-encode lines per MPRAGE inversion shot.

    Returns
    -------
    mask    : Boolean (size_z × size_y) sampling mask.
    samples : (N × 2) int array of [ky_offset, kz_offset] coordinates
              centred at (0, 0).
    n_calib : Number of calibration-region samples.
    n_random: Number of randomly drawn samples outside the calibration region.
    """
    np.random.seed(rand_seed)

    cy = (size_y + 1) / 2.0
    cz = (size_z + 1) / 2.0

    # Grid of 1-based indices
    Y, Z = np.meshgrid(np.arange(1, size_y + 1), np.arange(1, size_z + 1))

    # Normalised squared radius for the full k-space ellipse
    a, b = size_y / 2.0, size_z / 2.0
    R2 = ((Y - cy) / a) ** 2 + ((Z - cz) / b) ** 2

    # ── Elliptical shutter ────────────────────────────────────────────────────
    sample_mask = np.ones((size_z, size_y), dtype=bool)
    if elliptical:
        sample_mask = R2 <= 1.0

    # ── Calibration region (fully sampled ellipse at the centre) ──────────────
    ry, rz = calib_reg_y / 2.0, calib_reg_z / 2.0
    R2_calib = ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    mask_calib = (R2_calib <= 1.0) & sample_mask

    # ── Sample-count targets ──────────────────────────────────────────────────
    n_full   = size_y * size_z
    n_target = round(n_full / accel_factor)
    n_calib  = int(np.count_nonzero(mask_calib))

    # Round total acquired lines to a multiple of shot_length
    n_acq = round(n_target / shot_length) * shot_length
    n_acq = max(n_acq, n_calib)
    n_random = n_acq - n_calib

    # ── Variable-density weights ──────────────────────────────────────────────
    vd = np.ones((size_z, size_y), dtype=float)
    if variable_density > 0:
        R = np.sqrt(R2)
        vd = np.exp(-(R / 0.15) ** variable_density)
    vd[~sample_mask | mask_calib] = 0.0
    vd /= vd.sum()

    # ── Weighted random draw outside the calibration region ───────────────────
    valid_z, valid_y = np.where(sample_mask & ~mask_calib)
    weights = vd[sample_mask & ~mask_calib]
    drawn   = _weighted_sample_unique(len(weights), weights, n_random)

    # ── Assemble the final mask ───────────────────────────────────────────────
    mask = np.zeros((size_z, size_y), dtype=bool)
    mask[valid_z[drawn], valid_y[drawn]] = True
    mask[mask_calib] = True

    # Convert to centred (ky, kz) coordinate pairs
    z_idx, y_idx = np.where(mask)
    ky_coords = np.clip(y_idx - size_y // 2, -size_y // 2, size_y // 2 - 1)
    kz_coords = np.clip(z_idx - size_z // 2, -size_z // 2, size_z // 2 - 1)
    samples = np.stack([ky_coords, kz_coords], axis=1)   # shape (N, 2)

    return mask, samples, n_calib, n_random


def _build_shot_list(
    samples: np.ndarray,
    shot_length: int,
) -> list[np.ndarray]:
    """
    Order phase-encode samples into MPRAGE inversion-recovery shots.

    Samples are first sorted globally by polar angle (θ) then by radius (r),
    producing a roughly angular sweep.  Within each shot the lines are
    re-sorted by ascending radius so that the centre of k-space is acquired
    early in the inversion recovery.

    Parameters
    ----------
    samples     : (N × 2) array of [ky, kz] centred coordinates.
    shot_length : Lines per MPRAGE shot.

    Returns
    -------
    shot_list : List of 1-D index arrays, one per shot.
    """
    ky = samples[:, 0].astype(float)
    kz = samples[:, 1].astype(float)

    theta = np.arctan2(kz, ky)
    theta[theta < 0] += 2 * np.pi
    r = np.sqrt(ky ** 2 + kz ** 2)

    n_total = samples.shape[0]
    n_shots = int(np.ceil(n_total / shot_length))
    global_idx = np.lexsort((r, theta))   # primary: theta, secondary: r

    shot_list = []
    start = 0
    for _ in range(n_shots):
        stop = min(start + shot_length, n_total)
        sel  = global_idx[start:stop]
        sel  = sel[np.argsort(r[sel])]   # within-shot: ascending radius
        shot_list.append(sel)
        start = stop

    return shot_list


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_mprage_sampling_mask(
    kspace_3d: np.ndarray,
    size_of_center: tuple[int, int] = (32, 32),
    accel_factor: float = 4.0,
    variable_density: float = 0.8,
    elliptical_shutter: bool = True,
    mprage_shot_length: int = 64,
    rand_seed: int = 11235,
    verbose: bool = True,
) -> dict:
    """
    Generate a 3-D volumetric sampling mask for an MPRAGE acquisition.

    The readout dimension (kx, axis 0) is always **fully sampled**.
    Random undersampling is applied only in the two phase-encode dimensions
    (ky, axis 1) and (kz, axis 2).

    Parameters
    ----------
    kspace_3d          : 3-D numpy array with shape (kx, ky, kz).
                         Only the shape is used; the values are not read.
    size_of_center     : (ky_size, kz_size) of the fully-sampled calibration
                         ellipse at the centre of the PE plane.
    accel_factor       : Desired acceleration factor R ≥ 1.
    variable_density   : Density exponent (0 = uniform, >0 = centre-heavy).
                         Typical value: 0.8.
    elliptical_shutter : If True, restrict sampling to the inscribed ellipse
                         of the (ky, kz) plane.
    mprage_shot_length : Number of phase-encode lines per inversion-recovery
                         shot.  Automatically adjusted to divide kz evenly.
    rand_seed          : Random seed for reproducibility.
    verbose            : Print a summary to stdout.

    Returns
    -------
    result : dict with keys
        "mask_3d"          – Boolean ndarray, shape (kx, ky, kz).
                             True  → line is acquired.
                             False → line is not acquired.
        "mask_2d"          – Boolean ndarray, shape (ky, kz).
                             The underlying PE-plane sampling pattern.
        "samples"          – (N × 2) int array of centred [ky, kz] coords.
        "shot_list"        – List of index arrays (one per MPRAGE shot).
        "n_calib"          – Number of calibration-region PE lines.
        "n_random"         – Number of randomly drawn PE lines.
        "n_acquired"       – Total acquired PE lines.
        "accel_factor_eff" – Effective acceleration factor (after rounding).
        "shot_length"      – (Possibly adjusted) shot length used.
        "kspace_shape"     – (kx, ky, kz) shape tuple.
    """
    if kspace_3d.ndim != 3:
        raise ValueError(
            f"kspace_3d must be a 3-D array; got shape {kspace_3d.shape}."
        )

    n_kx, n_ky, n_kz = kspace_3d.shape

    # ── 1. Adjust shot length to divide kz evenly ─────────────────────────────
    shot_length = mprage_shot_length
    if n_kz % shot_length != 0:
        divisors = [d for d in range(4, n_kz + 1) if n_kz % d == 0]
        if divisors:
            idx = int(np.argmin(np.abs(np.array(divisors) - shot_length)))
            old_length = shot_length
            shot_length = divisors[idx]
            if verbose:
                print(
                    f"[INFO] Shot length adjusted {old_length} → {shot_length} "
                    f"to divide kz={n_kz} evenly."
                )
        else:
            if verbose:
                print(
                    f"[WARN] No divisor of kz={n_kz} found near {shot_length}; "
                    f"keeping original value."
                )

    # ── 2. Adjust acceleration factor for elliptical shutter ─────────────────
    if elliptical_shutter:
        elliptical_area = np.pi * (n_ky / 2.0) * (n_kz / 2.0)
        max_samples = int(np.floor(elliptical_area))
        min_af = (n_ky * n_kz) / max_samples
        if accel_factor < min_af:
            old_af = accel_factor
            accel_factor = np.ceil(min_af * 100) / 100
            if verbose:
                print(
                    f"[INFO] Acceleration factor adjusted {old_af:.4f} → "
                    f"{accel_factor:.4f} due to elliptical shutter constraint."
                )

    # ── 3. Generate 2-D PE-plane pattern ─────────────────────────────────────
    mask_2d, samples, n_calib, n_random = _poisson_pattern(
        size_y=n_ky,
        size_z=n_kz,
        variable_density=variable_density,
        accel_factor=accel_factor,
        elliptical=elliptical_shutter,
        rand_seed=rand_seed,
        calib_reg_y=size_of_center[0],
        calib_reg_z=size_of_center[1],
        shot_length=shot_length,
    )

    # ── 4. Build MPRAGE shot ordering ─────────────────────────────────────────
    shot_list = _build_shot_list(samples, shot_length)

    # ── 5. Expand 2-D mask to 3-D  (kx is fully sampled) ─────────────────────
    # mask_2d has shape (kz, ky); we need (ky, kz) then broadcast over kx
    pe_mask_ky_kz = mask_2d.T                               # (ky, kz)
    mask_3d = np.broadcast_to(
        pe_mask_ky_kz[np.newaxis, :, :],                    # (1, ky, kz)
        (n_kx, n_ky, n_kz),
    ).copy()                                                 # make writable

    # ── 6. Compute summary statistics ────────────────────────────────────────
    n_acquired   = n_calib + n_random
    n_full_pe    = n_ky * n_kz
    accel_factor_eff = n_full_pe / n_acquired

    ky_min, kz_min = samples.min(axis=0)
    ky_max, kz_max = samples.max(axis=0)

    if verbose:
        print("\n─────────────────── k-space summary ───────────────────")
        print(f"  Input k-space shape (kx, ky, kz) : {n_kx} × {n_ky} × {n_kz}")
        print(f"  Total Cartesian PE points         : {n_full_pe}")
        print(f"  Calibration-region samples        : {n_calib}")
        print(f"  Random samples                    : {n_random}")
        print(f"  Total acquired PE lines           : {n_acquired}")
        print(f"  Requested acceleration factor     : {accel_factor:.4f}")
        print(f"  Effective acceleration factor     : {accel_factor_eff:.4f}")
        print(f"  ky range                          : {ky_min} → {ky_max}")
        print(f"  kz range                          : {kz_min} → {kz_max}")
        print(f"  MPRAGE shot length                : {shot_length}")
        print(f"  Number of shots                   : {len(shot_list)}")
        print(f"  3-D mask shape                    : {mask_3d.shape}")
        print(f"  3-D mask density                  : "
              f"{mask_3d.mean()*100:.2f}% acquired")
        print("────────────────────────────────────────────────────────\n")

    return {
        "mask_3d":          mask_3d,
        "mask_2d":          pe_mask_ky_kz,
        "samples":          samples,
        "shot_list":        shot_list,
        "n_calib":          n_calib,
        "n_random":         n_random,
        "n_acquired":       n_acquired,
        "accel_factor_eff": accel_factor_eff,
        "shot_length":      shot_length,
        "kspace_shape":     (n_kx, n_ky, n_kz),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo  (run as a script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate a 3-D k-space volume (values are irrelevant – only shape is used)
    kspace = np.zeros((256, 192, 192), dtype=complex)

    result = generate_mprage_sampling_mask(
        kspace_3d          = kspace,
        size_of_center     = (32, 32),
        accel_factor       = 4.0,
        variable_density   = 0.8,
        elliptical_shutter = True,
        mprage_shot_length = 64,
        rand_seed          = 11235,
        verbose            = True,
    )

    mask_3d = result["mask_3d"]    # (256, 192, 192) bool
    mask_2d = result["mask_2d"]    # (192, 192) bool  – the PE pattern

    # ── Visualise the PE-plane sampling pattern ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a1a")

    ax = axes[0]
    ax.imshow(mask_2d.T, cmap="gray", origin="lower",
              extent=[-96, 96, -96, 96])
    ax.set_title(
        f"PE-plane sampling mask\n"
        f"R_eff = {result['accel_factor_eff']:.2f}  |  "
        f"N = {result['n_acquired']}  |  "
        f"shot length = {result['shot_length']}",
        color="white", fontsize=11,
    )
    ax.set_xlabel("ky", color="white")
    ax.set_ylabel("kz", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax = axes[1]
    ax.imshow(mask_3d[mask_3d.shape[0] // 2], cmap="gray",
              origin="lower", extent=[-96, 96, -96, 96])
    ax.set_title(
        f"Central kx slice of 3-D mask\nshape = {mask_3d.shape}",
        color="white", fontsize=11,
    )
    ax.set_xlabel("ky", color="white")
    ax.set_ylabel("kz", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout()
    plt.savefig("mprage_mask_preview.png", dpi=150, facecolor="#1a1a1a")
    plt.show()
    print("Preview saved → mprage_mask_preview.png")