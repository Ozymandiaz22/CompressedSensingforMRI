import numpy as np

def text_to_mask(filename):
    """Read a sampling list file and return a 256x256 mask with 1 for white
    (sampled) points and 0 for black points.

    filename: path to the text file. First two lines are integers and then
    lines of "ky,kz" with values in range -128..127.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("File must contain at least two header lines")

    # header values (kept for compatibility)
    try:
        num_samples = int(lines[0].strip())
        num_samples_high = int(lines[1].strip())
    except Exception:
        raise ValueError("First two lines must be integers")

    lines = lines[2:]

    mask = np.zeros((256, 256), dtype=np.uint8)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        ky, kz = parts[0].strip(), parts[1].strip()
        try:
            ky_i = int(ky) + 128
            kz_i = int(kz) + 128
        except ValueError:
            continue
        if 0 <= ky_i < 256 and 0 <= kz_i < 256:
            mask[kz_i, ky_i] = 1

    return mask

if __name__ == "__main__":
    # simple test when run as script
    import sys
    if len(sys.argv) > 1:
        out = text_to_mask(sys.argv[1])
        print(out.dtype, out.shape, out.sum())
    else:
        print("Provide a filename as argument")
