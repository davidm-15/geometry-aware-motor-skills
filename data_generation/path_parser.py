import numpy as np
from pathlib import Path
from utils.math_utils import euler_to_quaternion

# python -m data_generation.path_parser

def get_segments(positions, slice_size=7, threshold=0.005):
    if len(positions) < slice_size:
        return ['straight'] * len(positions)
        
    straight = []
    for i in range(len(positions) - slice_size + 1):
        pts = positions[i:i+slice_size]
        centroid = pts.mean(axis=0)
        centered_pts = pts - centroid
        
        if np.all(centered_pts == 0):
            straight.append(True)
            continue
            
        _, _, vh = np.linalg.svd(centered_pts)
        direction = vh[0]
        
        dist_sq = np.sum(centered_pts**2) - np.sum(np.dot(centered_pts, direction)**2)
        straight.append(dist_sq < threshold)
        
    pad_width = slice_size // 2
    rem = len(positions) - len(straight) - pad_width
    straight_padded = [straight[0]] * pad_width + straight + [straight[-1]] * max(0, rem)
    straight_padded = straight_padded[:len(positions)]
    
    return ['straight' if s else 'corner' for s in straight_padded]

def parse_trajectory_file(file_path):
    """
    Reads MaskPlanner dataset format with columns: X; Y; Z; A; B; C; strokeId
    """
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    
    # Group by strokeId
    stroke_ids = np.unique(data[:, -1])
    subpaths = []
    
    for sid in stroke_ids:
        stroke_data = data[data[:, -1] == sid]
        positions = stroke_data[:, 0:3]
        eulers = stroke_data[:, 3:6]
        quats = euler_to_quaternion(eulers)
        
        # Determine segments by curvature (simple heuristic)
        # Using a sliding window to identify straight vs corner
        segments = get_segments(positions)
        
        subpaths.append({
            'stroke_id': sid,
            'raw_data': stroke_data,
            'positions': positions,
            'quaternions': quats,
            'segments': segments,
        })
        
    return subpaths



if __name__ == "__main__":
    # Example usage
    L_shape = "datasets/L_shape/1_L_shape/trajectory.txt"
    window = "datasets/windows-v2/1_wr1fr_1/trajectory.txt"
    file_path = Path(L_shape)
    subpaths = parse_trajectory_file(file_path)
    
    for subpath in subpaths:
        print(f"Stroke ID: {subpath['stroke_id']}")
        print(f"Segments: {subpath['segments']}")  # Print first 10 segments for brevity