import numpy as np
import cv2
from  matplotlib import pyplot as plt
import scipy
import colorsys

# For line interpolation:
# Comment this out if not using 'get_interp_points' function
from bresenham import bresenham

def hsv_to_bgr(h, s, v):
    # Get RGB values
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 1/3:
        r, g, b = x, c, 0
    elif h < 0.5:
        r, g, b = 0, c, x
    elif h < 2/3:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # Scale RGB values to 0-255 range and convert to integers
    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return (b, g, r)


def get_distinct_colors(n):
    """
    Generate n visually distinct colors in BGR format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB; colorsys returns values between 0 and 1.
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        # Convert RGB to BGR in the range 0-255 for OpenCV.
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def is_color(img):
    if img.ndim <=2:
        return False

    # img.ndim >=3
    if img.shape[-1] == 1:
        return False

    return True
    

def show_img(img, color='gray', is_bgr=False, title='', figsize=None, final_show=True):
    """Show image using plt."""
    

    if figsize:
        if not isinstance(figsize, (tuple, list)):
            figsize = (figsize, figsize)
        plt.figure(figsize=figsize)

    if is_color(img) and is_bgr:
        img = img[...,::-1]

    params = {'cmap':color}

    if img.dtype == np.uint8:
        params.update({'vmin': 0, 'vmax': 255})

    plt.imshow(img, **params)
    plt.title(title)
    if final_show:
        plt.show()
    
    return


def draw_xrange(img, xrange):
    annot_img = img.copy()
    im_h, im_w = annot_img.shape[:2]
    annot_img = cv2.line(annot_img, (xrange[0], 0), (xrange[0], im_h), (0,0,255), thickness=1)
    annot_img = cv2.line(annot_img, (xrange[1], 0), (xrange[1], im_h), (0,0,255), thickness=1)
    return annot_img

def get_xrange(bin_line_mask):
    """
        bin_line_mask: np.ndarray => black and white binary mask of line
        black => background => 0
        white => foregrond line pixel => 255
        returns: (x_start, x_end) where x_start and x_end represent the starting and ending points 
                for the binary line segment
    """
    # print(bin_line_mask.sum(axis=0))
    # np.save("problem_mask.npy", bin_line_mask)
    smooth_signal = scipy.signal.medfilt(bin_line_mask.sum(axis=0), kernel_size=5)
    # print(smooth_signal.shape)
    # print(smooth_signal)
    # print(np.nonzero(smooth_signal))
    x_range = np.nonzero(smooth_signal)
    if len(x_range) and len(x_range[0]): # To handle cases with empty masks
        x_range = x_range[0][[0, -1]]
    else:
        x_range = None
    return x_range

def get_kp(line_img, interval=10, x_range=None, get_num_lines=False, get_center=True, multi_kp=False, vert_gap_thresh=2):
    """
    Extract keypoints from a binary mask of a line.
    """
    im_h, im_w = line_img.shape[:2]
    kps = []
    if x_range is None:
        x_range = (0, im_w)
    
    # Track the number of vertical components in each x-column
    num_comps = []
    
    for x in range(x_range[0], x_range[1], interval):
        # Get all white pixels in this column
        all_y_points = np.where(line_img[:, x] == 255)[0]
        
        if len(all_y_points) == 0:
            continue
            
        # Group points into components based on vertical gaps
        components = []
        current_component = [all_y_points[0]]
        
        for y in all_y_points[1:]:
            if y - current_component[-1] > vert_gap_thresh:
                # Gap detected, start new component
                components.append(current_component)
                current_component = [y]
            else:
                current_component.append(y)
        
        components.append(current_component)
        
        # Add keypoints for each component
        if multi_kp:
            for comp in components:
                if get_center:
                    y = int(np.mean(comp))
                else:
                    y = int(comp[0])  # or any other strategy
                kps.append({"x": float(x), "y": float(y)})
        else:
            if len(components) == 1:
                y = int(np.mean(components[0])) if get_center else int(components[0][0])
                kps.append({"x": float(x), "y": float(y)})
        
        num_comps.append(len(components))

    if get_num_lines:
        return kps, int(np.median(num_comps)) if num_comps else 1
    return kps


def draw_edge(img, edge):
    inter_points = get_interp_points(edge[0], edge[1])
    # print(inter_points)
    # print(len(inter_points), inter_points)
    annot_img = draw_kps(img, array_to_points(inter_points), color=(255,0,0))
    return annot_img

def draw_kps(img, kps, color=(0,255,0), classes=None, **draw_options):
    if is_color(img):
        annot_img = img.copy()
    else:
        annot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if classes is None:
        classes = [0]*len(kps)
        color_map = {0: color}
    else:
        colors = list(get_distinct_colors(classes.max()+1))
        color_map = dict(zip(range(classes.max()+1), colors))
        # print(color_map)
    for idx, kp in enumerate(kps):
        options = dict(color=color_map[classes[idx]], markerType=cv2.MARKER_CROSS, markerSize=2, thickness=2, line_type=8)
        options.update(draw_options)
        annot_img = cv2.drawMarker(annot_img, (int(kp['x']), int(kp['y'])), **options)
    return annot_img

def points_to_array(pred_ds):
    res = []
    for line in pred_ds:
        line_arr = []
        for pt in line:
            line_arr.append([pt['x'], pt['y']])
        res.append(line_arr)
    return res

# res = line_utils.draw_lines(img, points_to_array(pred_ds))
def draw_lines(img, lines, thickness=2):
    """
    Draw digitized lines on an image based on keypoints.
    
    Args:
        img (numpy.ndarray): Input image.
        lines (list): A list of lines, where each line is a list of keypoints.
                      Each keypoint can be a dict with keys "x" and "y" or a list/tuple.
        thickness (int): Thickness for the drawn lines.
    
    Returns:
        numpy.ndarray: The image with lines drawn.
    """
    annotated_img = img.copy()
    
    # Wrap a single line into a list if needed.
    if lines and (isinstance(lines[0], dict) or 
                  (isinstance(lines[0], (list, tuple)) and len(lines[0]) == 2 and isinstance(lines[0][0], (int, float)))):
        if not (isinstance(lines[0][0], (list, dict))):
            lines = [lines]
    
    n_lines = len(lines)
    # Generate distinct colors if we have more than one line.
    if n_lines > 1:
        colors = get_distinct_colors(n_lines)
    else:
        colors = [(0, 255, 0)]
    
    for idx, line in enumerate(lines):
        pts = []
        for pt in line:
            if isinstance(pt, dict):
                x_val = int(round(pt["x"]))
                y_val = int(round(pt["y"]))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x_val = int(round(pt[0]))
                y_val = int(round(pt[1]))
            else:
                continue
            pts.append([x_val, y_val])
        if pts:
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            annotated_img = cv2.polylines(annotated_img, [pts], isClosed=False, color=colors[idx], thickness=thickness)
    
    return annotated_img


# Get the line points that would lie between ptA and ptB, according to the bresenham algorithm
def get_interp_points(ptA, ptB, thickness=1):
    # x_interp = np.arange(ptA[0], ptB[0])
    # y_interp = np.interp(x_interp, [ptA[0], ptB[0]], [ptA[1], ptB[1]]).round().astype(int)
    
    points = []
    delta_range = (-thickness//2, thickness//2)

    for delta in range(delta_range[0], delta_range[1]+1):
        points.extend(list(bresenham(ptA[0], ptA[1]+delta, ptB[0], ptB[1]+delta)))

    inter_points = np.array(points)
    
    # inter_points = np.stack([x_interp,y_interp], axis=-1)
    return inter_points

def array_to_points(pts_arr):
    pts = [{'x': pt[0], 'y': pt[1]} for pt in pts_arr]
    return pts