import numpy as np
import struct

def mread(datapath, expt_to_show=None):
    """
    Python translation of MR Solutions mread3d.m
    Reads .MRD files from MR Solutions scanners.

    Parameters
    ----------
    datapath : str
        Path to the .MRD or .SUR file
    expt_to_show : int, optional
        Override number of experiments to read (useful for partial files)

    Returns
    -------
    data : dict with keys:
        path        - file path
        header      - raw 256 int16 header values
        samples     - number of samples per view
        views       - number of view lines per slice
        sliceviews  - number of views in slice direction (3D)
        slices      - number of slices
        type        - data type code (19 = complex int16, 21 = complex float32)
        echoes      - number of echoes per shot
        nex         - number of experiments
        parameters  - ASCII parameter footer as string
        data        - numpy array shape (samples, views, sliceviews, slices, echoes, nex)
                      dtype is complex128 for complex data, float64 for real data
    """

    data = {}
    data['path'] = datapath

    with open(datapath, 'rb') as fid:

        # --- Read 256 int16 header values ---
        fid.seek(0)
        header = np.frombuffer(fid.read(256 * 2), dtype=np.int16)
        data['header'] = header

        # Header fields (1-indexed in MATLAB -> 0-indexed here)
        data['views']      = int(header[2])   # header(3)
        data['sliceviews'] = int(header[4])   # header(5)
        data['slices']     = int(header[6])   # header(7)
        data['type']       = int(header[9])   # header(10)
        data['echoes']     = int(header[76])  # header(77)
        data['nex']        = int(header[78])  # header(79)

        if expt_to_show is not None:
            data['nex'] = int(expt_to_show)

        # Number of samples is stored as int32 at start of file
        fid.seek(0)
        data['samples'] = struct.unpack('<i', fid.read(4))[0]

        # --- Calculate total points ---
        totpts = (data['samples'] * data['views'] * data['sliceviews'] *
                  data['slices'] * data['echoes'] * data['nex'])

        # --- Data type map: (numpy dtype, bytes per value, is_complex) ---
        dtype_map = {
            3:  (np.int16,   2, False),  # real int16
            16: (np.uint8,   1, True),   # complex uint8
            17: (np.int8,    1, True),   # complex int8
            18: (np.int16,   2, True),   # complex int16
            19: (np.int16,   2, True),   # complex int16
            20: (np.int32,   4, True),   # complex int32
            21: (np.float32, 4, True),   # complex float32
            22: (np.float64, 8, True),   # complex float64
        }

        if data['type'] not in dtype_map:
            raise ValueError(f"Unhandled data type: {data['type']}")

        dtype_np, ptsize, is_complex = dtype_map[data['type']]

        # For complex data, file stores 2*totpts values (interleaved real/imag)
        multiplier = 2 if is_complex else 1

        # --- Read raw bytes as numeric array (NOT complex yet) ---
        fid.seek(512)
        rawdata = np.frombuffer(
            fid.read(multiplier * totpts * ptsize),
            dtype=np.dtype(dtype_np).newbyteorder('<')
        )

        # --- Read ASCII parameter footer ---
        footer_offset = 512 + multiplier * totpts * ptsize
        fid.seek(footer_offset)
        data['parameters'] = fid.read().decode('latin-1', errors='replace')

    # --- Combine interleaved real/imaginary pairs into complex array ---
    # File layout: [re0, im0, re1, im1, re2, im2, ...]
    if is_complex:
        rawdata = (rawdata[0::2].astype(np.float64)
                   + 1j * rawdata[1::2].astype(np.float64))
    else:
        rawdata = rawdata.astype(np.float64)

    # --- Reshape to (samples, views, sliceviews, slices, echoes, nex) ---
    # order='F' matches MATLAB's column-major memory layout
    rawdata = rawdata[:totpts].reshape(
        data['samples'],
        data['views'],
        data['sliceviews'],
        data['slices'],
        data['echoes'],
        data['nex'],
        order='F'
    )

    data['data'] = rawdata
    return data


# --- Example usage ---
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'yourfile.MRD'
    mrd = mread(path)

    print(f"Samples:    {mrd['samples']}")
    print(f"Views:      {mrd['views']}")
    print(f"Sliceviews: {mrd['sliceviews']}")
    print(f"Slices:     {mrd['slices']}")
    print(f"Echoes:     {mrd['echoes']}")
    print(f"NEX:        {mrd['nex']}")
    print(f"Type:       {mrd['type']}")
    print(f"Data shape: {mrd['data'].shape}")
    print(f"Data dtype: {mrd['data'].dtype}")
    print(f"Is complex: {np.iscomplexobj(mrd['data'])}")

    if np.iscomplexobj(mrd['data']):
        print(f"Real range: [{mrd['data'].real.min():.3f}, {mrd['data'].real.max():.3f}]")
        print(f"Imag range: [{mrd['data'].imag.min():.3f}, {mrd['data'].imag.max():.3f}]")