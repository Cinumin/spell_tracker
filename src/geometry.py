import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def dtw_distance(seq_a, seq_b):
    if len(seq_a) == 0 or len(seq_b) == 0:
        raise ValueError("Both sequences must be non-empty.")
    cost_matrix = cdist(seq_a, seq_b, metric='euclidean') # compute pairwise distances between points in seq_a and seq_b
    n, m = len(seq_a), len(seq_b) # construct nxm cost matrix
    D = np.full((n + 1, m + 1), np.inf) # initialize with infinity
    D[0, 0] = 0 # starting point
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1] 
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]

def resample_strokes(stroke, control_points):
    if not stroke:
        raise ValueError("Stroke must be non-empty.")
    if control_points <= 0:
        raise ValueError("Control points must be a positive integer.")

    points = np.array(stroke) # convert stroke to numpy array
    seg_length = np.linalg.norm(np.diff(stroke, axis=0), axis=1) # compute magnitude of each segment; axis=0 means compute along rows, axis=1 means compute along columns
    cumulative_length = np.concatenate(([0], np.cumsum(seg_length))) # cumulative length along the stroke
    total = cumulative_length[-1] # total length of the stroke
    t = np.linspace(0, total, control_points) # create control_points evenly spaced points along the total length

    resampled_stroke = []
    indices =np.searchsorted(cumulative_length, t, side='right') - 1 # find the indices of the segments that contain each control point
    indices = np.clip(indices, 0, len(points) - 2) # ensure indices are within bounds
    for i, indice in enumerate(indices):
        fraction =(t[i] - cumulative_length[indice]) / (cumulative_length[indice + 1] - cumulative_length[indice]) # compute the relative position of each control point within its segment
        resampled_point = points[indice] + fraction * (points[indice + 1] - points[indice]) # linear interpolation to find the resampled point
        resampled_stroke.append(resampled_point)
    return resampled_stroke

def normalize_strokes(strokes):
    """Translate and uniformly scale strokes into a centered unit square.

    Uses the combined bounding box across all strokes (not per-stroke) so
    relative stroke sizes and positions to each other are preserved.
    Coordinate-convention-agnostic: callers needing a y-flip (e.g.
    reference data) must do it before calling this.

    Returns a list of (n_i, 2) float arrays, one per input stroke, in the
    same order with the same per-stroke point counts.
    """
    arrays = [np.asarray(stroke, dtype=float) for stroke in strokes]
    if not arrays or all(len(a) == 0 for a in arrays):
        raise ValueError("no points to normalize")

    combined = np.concatenate(arrays) # compute combined bounding box
    min_xy = combined.min(axis=0)  # find min x and y across all points for column 0 and column 1
    max_xy = combined.max(axis=0) # find max x and y across all points for column 0 and column 1
    width, height = max_xy - min_xy
    span = max(width, height) # uniform scale factor to fit into a unit square, preserving aspect ratio
    scale = 1.0 / span if span > 0 else 1.0 

    offset = np.array([(1 - width * scale) / 2, (1 - height * scale) / 2])
    return [(arr - min_xy) * scale + offset for arr in arrays]


def load_reference(character, path='graphics.txt'):
    """Load a character's reference median strokes from a Make Me a Hanzi
    graphics.txt-style file (one JSON object per line, matched by the
    "character" field), y-flipped to top-down image coordinates.

    Make Me a Hanzi's medians use a y-up 1024-wide box with the logical
    origin at (0, 900); flipping via y = 900 - y_raw matches the top-down
    (y increases downward) convention hand_tracker.py's pixel coordinates
    use, so the two can be compared directly after normalize_strokes.

    Returns the medians for `character`, in drawing order, not yet
    normalized.
    """
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if data['character'] == character:
                return [[(x, 900 - y) for x, y in stroke] for stroke in data['medians']]
    raise ValueError(f"character {character!r} not found in {path}")


def compare_to_reference(user_strokes, reference_medians):
    """Compare user-drawn strokes against reference median strokes.

    Strokes are paired up in drawing order (1st user stroke vs. 1st
    reference stroke, etc.); both sets are independently normalized (own
    combined bounding box, uniform scale, centered) so absolute
    position/scale/webcam-frame-size differences don't affect the score,
    then each pair is compared with dtw_distance.

    Returns (total_distance, per_stroke_distances), where total_distance
    is the sum of per-stroke distances -- summing lets total_distance
    scale with total drawing "effort"/length rather than have a single
    badly-drawn stroke averaged away by several well-drawn ones.

    Raises ValueError if the stroke counts differ, since drawing the
    wrong number of strokes is itself meaningful feedback rather than
    something to silently truncate or pad around.
    """
    if len(user_strokes) != len(reference_medians):
        raise ValueError(
            f"expected {len(reference_medians)} strokes, got {len(user_strokes)}"
        )

    normalized_user = normalize_strokes(user_strokes)
    normalized_ref = normalize_strokes(reference_medians)
    cost_matrix = np.array([[dtw_distance(u, r) for r in normalized_ref] for u in normalized_user])
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    per_stroke_distances = cost_matrix[row_indices, col_indices]
    return sum(per_stroke_distances), per_stroke_distances


if __name__ == "__main__":
    medians = load_reference('火', path='huo.json')
    for stroke in normalize_strokes(medians):
        print(stroke)
