import numpy as np
import pydicom

# --- Load DICOM ---
ds = pydicom.dcmread("pet_study")

# --- Extract basic dimensions ---
num_frames = int(ds[(0x0028, 0x0008)].value)
rows = int(ds[(0x0028, 0x0010)].value)
cols = int(ds[(0x0028, 0x0011)].value)

# --- Raw pixel data ---
pixel_array = ds.pixel_array  # (num_frames, rows, cols)

# --- Extract metadata ---
frame_positions = np.array(ds[(0x0055, 0x1002)].value)  # Z
print(len(frame_positions))
print(num_frames)
frame_times = np.array(ds[(0x0055, 0x1001)].value)  # time
frame_durations = np.array(ds[(0x0055, 0x1004)].value)  # duration

print("=== BASIC DIMENSIONS ===")
print("num_frames:", num_frames)
print("pixel_array shape:", pixel_array.shape)

assert pixel_array.shape[0] == num_frames, "Pixel array mismatch"

# =========================
# FRAME POSITION HANDLING
# =========================

print("\n=== FRAME POSITION CHECK ===")

frame_positions_raw = np.array(ds[(0x0055, 0x1002)].value)

print("raw frame_positions length:", len(frame_positions_raw))

# Expect either:
# - num_frames (already Z)
# - num_frames * 3 (XYZ per frame)

if len(frame_positions_raw) == num_frames * 3:
    print("Detected XYZ triplets per frame → reshaping")
    frame_positions = frame_positions_raw.reshape(num_frames, 3)
    frame_z = frame_positions[:, 2]
elif len(frame_positions_raw) == num_frames:
    print("Detected scalar per frame → using directly as Z")
    frame_z = frame_positions_raw
else:
    raise ValueError("Unexpected frame_positions length")

print("frame_z shape:", frame_z.shape)

# =========================
# TIME VECTOR CHECK
# =========================

print("\n=== TIME VECTOR CHECK ===")

frame_times = np.array(ds[(0x0055, 0x1001)].value)
frame_durations = np.array(ds[(0x0055, 0x1004)].value)

print("frame_times length:", len(frame_times))
print("frame_durations length:", len(frame_durations))

T = len(frame_times)

assert len(frame_durations) == T, "Durations mismatch"

# =========================
# GRID CONSISTENCY
# =========================

print("\n=== GRID CONSISTENCY ===")

unique_z = np.unique(frame_z)
Z = len(unique_z)

print("Unique Z slices:", Z)
print("Time steps (T):", T)

assert num_frames == T * Z, "Frames != T * Z → layout assumption broken"

# =========================
# BLOCK STRUCTURE CHECK
# =========================

print("\n=== BLOCK STRUCTURE CHECK ===")

for t in range(T):
    start = t * Z
    end = (t + 1) * Z
    z_block = frame_z[start:end]

    if len(np.unique(z_block)) != Z:
        raise ValueError(f"Time step {t} missing slices")

print("All time steps contain full Z stacks")

# =========================
# Z CONSISTENCY ACROSS TIME
# =========================

print("\n=== Z CONSISTENCY ACROSS TIME ===")

reference_sorted = np.sort(frame_z[:Z])

for t in range(1, T):
    start = t * Z
    end = (t + 1) * Z

    if not np.allclose(np.sort(frame_z[start:end]), reference_sorted):
        raise ValueError(f"Inconsistent Z set at time {t}")

print("All time steps share identical Z positions")

# =========================
# OPTIONAL: ORDER CHECK
# =========================

print("\n=== ORDER CHECK (optional) ===")

first_block = frame_z[:Z]

if np.all(np.diff(first_block) > 0):
    print("Z already sorted ascending within blocks")
elif np.all(np.diff(first_block) < 0):
    print("Z sorted descending within blocks")
else:
    print("Z not sorted → will require sorting")

# =========================
# FINAL SUMMARY
# =========================

print("\n=== SUMMARY ===")
print(f"T = {T}, Z = {Z}, total frames = {num_frames}")
print("All sanity checks passed") == Z, f"Time {t} is missing slices"
