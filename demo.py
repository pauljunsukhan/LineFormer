import infer
import cv2
import line_utils
import os
from datetime import datetime
import mmcv  # used to load annotation files if cleaning is enabled
from scipy.interpolate import CubicSpline  # used for cubic interpolation
import numpy as np
import scipy.signal

# ============================================================
# KNOBS: Adjust these as needed.
# ============================================================
use_cleaning = False               # Enable cleaning if your image has background noise/misalignment.
annot_file = "demo/LTC5959982___3_HTML.json" if use_cleaning else None

use_post_processing = True         # Enable extra mask post-processing (e.g., morphological operations).
interp_method = "cubic"             # Choose interpolation method: "linear" or "cubic"
mask_kp_sample_interval = 1         # Keypoint sampling interval along the predicted line mask.

# NEW KNOBS:
mask_score_threshold = 0.2          # New knob: Adjust the instance detection score threshold.
extract_multiple_kps = True         # New knob: Extract all keypoints in a column—even if multiple peaks overlap.
vertical_gap_threshold = 1          # New knob: The minimum vertical pixel gap to consider two white pixels as separate components.

# ============================================================
# Monkey-patch the interpolation function if available in infer.
# ============================================================
if hasattr(infer, "interpolate"):
    old_interpolate = infer.interpolate

    def new_interpolate(line_ds, inter_type="linear"):
        # Always use our knob value instead of the passed-in inter_type
        inter_type = interp_method
        
        # Determine if input is a flat list of keypoints or a list-of-lines.
        if line_ds and isinstance(line_ds[0], dict):
            line_ds_nested = [line_ds]
            single_line = True
        elif line_ds and isinstance(line_ds[0], list):
            line_ds_nested = line_ds
            single_line = False
        else:
            line_ds_nested = line_ds
            single_line = False

        if inter_type.lower() == "cubic":
            # Custom cubic spline interpolation applied to each line.
            inter_line_ds = []
            for ds in line_ds_nested:
                if len(ds) < 2:
                    inter_line_ds.append(ds)
                    continue
                new_ds = []
                for pt in ds:
                    if isinstance(pt, dict):
                        new_ds.append(pt)
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        new_ds.append({"x": pt[0], "y": pt[1]})
                    else:
                        new_ds.append(pt)
                # Sort the keypoints by x.
                new_ds_sorted = sorted(new_ds, key=lambda pt: pt["x"])
                xs = [pt["x"] for pt in new_ds_sorted]
                ys = [pt["y"] for pt in new_ds_sorted]
                # Remove duplicate or non-increasing x values.
                unique_x = []
                unique_y = []
                for x_val, y_val in zip(xs, ys):
                    if not unique_x or x_val > unique_x[-1]:
                        unique_x.append(x_val)
                        unique_y.append(y_val)
                if len(unique_x) < 2:
                    inter_line_ds.append(ds)
                    continue

                cs = CubicSpline(unique_x, unique_y)
                x_min = int(unique_x[0])
                x_max = int(unique_x[-1])
                new_line = [{"x": x, "y": int(cs(x))} for x in range(x_min, x_max + 1)]
                inter_line_ds.append(new_line)
            return inter_line_ds[0] if single_line else inter_line_ds
        else:
            # Linear interpolation: process each line separately.
            inter_line_ds = []
            for ds in line_ds_nested:
                new_ds = []
                for pt in ds:
                    if isinstance(pt, dict):
                        new_ds.append(pt)
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        new_ds.append({"x": pt[0], "y": pt[1]})
                    else:
                        new_ds.append(pt)
                # Use the original interpolate function for linear interpolation.
                inter_line = old_interpolate(new_ds, inter_type=inter_type)
                inter_line_ds.append(inter_line)
            return inter_line_ds[0] if single_line else inter_line_ds

    infer.interpolate = new_interpolate
else:
    print("Warning: interpolate function not found in infer module. Using default interpolation.")

# ============================================================
# Monkey-patch the instance detection threshold if available.
# ============================================================
if hasattr(infer, "do_instance"):
    old_do_instance = infer.do_instance

    def new_do_instance(model, img, score_thr=0.3):
        # Override the score threshold with our knob value.
        return old_do_instance(model, img, score_thr=mask_score_threshold)

    infer.do_instance = new_do_instance

# ============================================================
# Monkey-patch get_kp to extract multiple keypoints per column (if enabled).
# ============================================================
if extract_multiple_kps:
    # Save the original get_kp function from line_utils.
    original_get_kp = line_utils.get_kp

    def new_get_kp(*args, **kwargs):
        # Inject our knobs into every call:
        kwargs["multi_kp"] = extract_multiple_kps
        kwargs["vert_gap_thresh"] = vertical_gap_threshold
        return original_get_kp(*args, **kwargs)

    line_utils.get_kp = new_get_kp

# ============================================================
# Optionally load an annotation file for cleaning (if available)
# ============================================================
annotation = None
if use_cleaning and annot_file is not None:
    annotation = mmcv.load(annot_file)

# ============================================================
# Set the input image path and read the image.
# ============================================================
img_path = "demo/LTC3097_fig16_highres_cropped_again.png"
img = cv2.imread(img_path)  # BGR format

# ============================================================
# Define model parameters and load the model.
# ============================================================
CKPT = "iter_3000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cuda"  # run on GPU
infer.load_model(CONFIG, CKPT, DEVICE)

# ============================================================
# Extract the line dataseries.
# ============================================================
line_dataseries = infer.get_dataseries(
    img,
    annot=annotation,
    to_clean=use_cleaning,
    post_proc=use_post_processing,
    mask_kp_sample_interval=mask_kp_sample_interval,
    interp_method=interp_method
)

# ============================================================
# Draw digitized lines on the image using the detected keypoints.
# ============================================================
keypoints_array = line_utils.points_to_array(line_dataseries)
img = line_utils.draw_lines(img, keypoints_array)

# ============================================================
# Create output file name with knob settings appended (timestamp comes before knobs).
# ============================================================
base_name, ext = os.path.splitext(os.path.basename(img_path))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
knob_dict = {}
if use_cleaning:
    knob_dict["clean"] = 1
if use_post_processing:
    knob_dict["post"] = 1
if interp_method:
    knob_dict["interp"] = interp_method
if mask_kp_sample_interval:
    knob_dict["int"] = mask_kp_sample_interval
knob_dict["thr"] = mask_score_threshold
knob_dict["multi_kp"] = int(extract_multiple_kps)
knob_dict["vertgap"] = vertical_gap_threshold

knob_suffix = "_" + "_".join([f"{k}-{v}" for k, v in knob_dict.items()]) if knob_dict else ""
output_file = os.path.join(os.path.dirname(img_path), f"{base_name}_digitized_{timestamp}_digitized{knob_suffix}{ext}")

# ============================================================
# Write the output image.
# ============================================================
cv2.imwrite(output_file, img)

