import numpy as np
import pydicom


def load_dicom(path):
    return pydicom.dcmread(path)


def extract_basic(ds):
    num_frames = int(ds[(0x0028, 0x0008)].value)
    rows = int(ds[(0x0028, 0x0010)].value)
    cols = int(ds[(0x0028, 0x0011)].value)
    pixel_array = ds.pixel_array
    return num_frames, rows, cols, pixel_array


def extract_frame_metadata(ds, num_frames):
    frame_times = np.array(ds[(0x0055, 0x1001)].value)
    frame_durations = np.array(ds[(0x0055, 0x1004)].value)

    frame_positions_raw = np.array(ds[(0x0055, 0x1002)].value)
    if len(frame_positions_raw) == num_frames * 3:
        frame_positions = frame_positions_raw.reshape(num_frames, 3)
        frame_z = frame_positions[:, 2]
    elif len(frame_positions_raw) == num_frames:
        frame_z = frame_positions_raw
    else:
        raise ValueError("Unexpected frame_positions length")

    return frame_times, frame_durations, frame_z


def extract_spacing(ds):
    pixel_spacing = np.array(ds[(0x0028, 0x0030)].value, dtype=float)
    slice_spacing = float(ds[(0x0018, 0x0088)].value)

    row_spacing, col_spacing = pixel_spacing
    z_spacing = slice_spacing
    return row_spacing, col_spacing, z_spacing


def build_volume_4d(pixel_array, T, Z, rows, cols):
    return pixel_array.reshape(T, Z, rows, cols)
