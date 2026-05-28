import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import argparse
import os


def measure_and_visualize_window(obj_path, cardboard_sides=None, thickness=None, output_path=None, output_obj=None):
    # 1. Load the 3D model
    print(f"Loading {obj_path}...")
    if not os.path.exists(obj_path):
        print(f"Error: File {obj_path} does not exist.")
        return
        
    mesh = trimesh.load(obj_path, force='mesh')
    
    # 2. Determine orientation and dimensions
    extents = mesh.extents
    bounds = mesh.bounds
    bounds_center = (bounds[0] + bounds[1]) / 2
    
    # Smallest dimension is depth
    axis_depth = np.argmin(extents)
    axes_2d = [i for i in range(3) if i != axis_depth]
    axis_x, axis_y = axes_2d[0], axes_2d[1]
    
    dim_x, dim_y = extents[axis_x], extents[axis_y]
    depth_raw = extents[axis_depth]
    
    scale = 1.0
    if cardboard_sides is not None:
        # Optimized scale to fit rectangular cardboard in either orientation
        # Orientation 1: dim_x fits side 0, dim_y fits side 1
        s1 = min(cardboard_sides[0] / dim_x, cardboard_sides[1] / dim_y)
        # Orientation 2: dim_x fits side 1, dim_y fits side 0
        s2 = min(cardboard_sides[1] / dim_x, cardboard_sides[0] / dim_y)
        
        scale = max(s1, s2)
        print(f"Applying optimal scale factor: {scale:.4f} (Fitting {dim_x:.1f}x{dim_y:.1f} window into {cardboard_sides[0]}x{cardboard_sides[1]} cardboard)")
    
    # 3. Extract 2D Profile for measurements and visualization
    normal = np.zeros(3)
    normal[axis_depth] = 1.0
    slice_3d = mesh.section(plane_origin=bounds_center, plane_normal=normal)

    # 4. Identify boundaries using Segment Density Profiling
    def find_boundaries_from_profile(slice_path, axis_idx):
        if slice_path is None: return []
        
        other_axis = axis_x if axis_idx == axis_y else axis_y
        total_span = extents[other_axis]
        
        # 4a. Collect axis-aligned segments (perpendicular to sweep axis)
        votes = []
        for entity in slice_path.entities:
            discrete = entity.discrete(slice_path.vertices)
            for i in range(len(discrete)-1):
                p1, p2 = discrete[i], discrete[i+1]
                # A boundary is a coordinate where many segments are orthogonal to the sweep
                if abs(p1[axis_idx] - p2[axis_idx]) < 0.1: # 0.1mm alignment tolerance
                    length = np.linalg.norm(p1 - p2)
                    votes.append((p1[axis_idx], length))
        
        if not votes: return []
        
        # 4b. Cluster votes within 2.0mm and sum their lengths
        votes.sort()
        clusters = []
        curr_coord, curr_weight = votes[0]
        for coord, weight in votes[1:]:
            if coord - curr_coord < 2.0:
                # Merge into current cluster (weighted position)
                curr_coord = (curr_coord * curr_weight + coord * weight) / (curr_weight + weight)
                curr_weight += weight
            else:
                clusters.append((curr_coord, curr_weight))
                curr_coord, curr_weight = coord, weight
        clusters.append((curr_coord, curr_weight))
        
        # 4c. Filter for MAJOR boundaries only
        # Keep only levels where the combined segments span at least 10% of the window width/height
        significant = [c[0] for c in clusters if c[1] > total_span * 0.1]
        
        return np.sort(significant)

    pts_x = find_boundaries_from_profile(slice_3d, axis_x)
    pts_y = find_boundaries_from_profile(slice_3d, axis_y)
    
    pts_x_sorted = np.sort(pts_x)
    pts_y_sorted = np.sort(pts_y)[::-1] # Top to Bottom

    def calculate_measurements(points, scale_factor):
        individual = []
        cumulative = []
        if len(points) < 2: return individual, cumulative
        
        start_pt = points[0]
        for i in range(len(points) - 1):
            length = abs(points[i+1] - points[i]) * scale_factor
            segment_type = "Solid" if i % 2 == 0 else "Gap"
            individual.append((segment_type, length))
            cumulative.append(abs(points[i+1] - start_pt) * scale_factor)
        return individual, cumulative

    ind_x, cum_x = calculate_measurements(pts_x_sorted, scale)
    ind_y, cum_y = calculate_measurements(pts_y_sorted, scale)

    # 5. OBJ Generation (if requested)
    if output_obj and slice_3d is not None and thickness is not None:
        print(f"Generating scaled OBJ with thickness {thickness}...")
        # to_planar converts Path3D to Path2D and returns a 4x4 transform
        slice_2d, transform_to_3d = slice_3d.to_planar()
        
        # Scale the 2D profile (X, Y dimensions)
        slice_2d.apply_scale(scale)
        
        # Extrude by the PROVIDED thickness (not scaled original depth)
        mesh_extruded = slice_2d.extrude(thickness)
        
        # Origin Customization: Offset from bottom-left corner
        # (Move origin 3.9 right, 2.6 up, and 1.75 into the thickness)
        min_bounds = mesh_extruded.bounds[0]
        target_origin = min_bounds + np.array([3.9, 2.6, 1.75])
        mesh_extruded.apply_translation(-target_origin)
        
        # Export
        mesh_extruded.export(output_obj)
        print(f"Saved cardboard CAD model to {output_obj}")

    # 7. Visualization
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')
    ax.axis('off')

    if slice_3d is not None:
        for entity in slice_3d.entities:
            discrete = entity.discrete(slice_3d.vertices)
            ax.plot(discrete[:, axis_x], discrete[:, axis_y], color='black', linewidth=1.5)

    x_min, x_max = bounds[0][axis_x], bounds[1][axis_x]
    y_min, y_max = bounds[0][axis_y], bounds[1][axis_y]
    
    pad_x = extents[axis_x] * 0.4
    pad_y = extents[axis_y] * 0.4
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # 7a. Annotate Horizontal (Blue)
    y_anno = y_min - (extents[axis_y] * 0.15)
    for i in range(len(pts_x_sorted) - 1):
        x1, x2 = pts_x_sorted[i], pts_x_sorted[i+1]
        mid_x = (x1 + x2) / 2
        seg_type, length = ind_x[i]
        ax.plot([x1, x2], [y_anno, y_anno], color='blue', linewidth=1.5)
        ax.plot([x1, x1], [y_min, y_anno - 10], color='blue', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.plot([x2, x2], [y_min, y_anno - 10], color='blue', alpha=0.3, linestyle='--', linewidth=0.8)
        arrow = FancyArrowPatch((x1, y_anno), (x2, y_anno), arrowstyle='<->', mutation_scale=15, color='blue')
        ax.add_patch(arrow)
        text_str = f"{seg_type}\n{length:.2f}\nΣ:{cum_x[i]:.2f}"
        ax.text(mid_x, y_anno - 5, text_str, ha='center', va='top', fontsize=10, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # 7b. Annotate Vertical (Red)
    x_anno = x_min - (extents[axis_x] * 0.15)
    for i in range(len(pts_y_sorted) - 1):
        y1, y2 = pts_y_sorted[i], pts_y_sorted[i+1]
        mid_y = (y1 + y2) / 2
        seg_type, length = ind_y[i]
        ax.plot([x_anno, x_anno], [y1, y2], color='red', linewidth=1.5)
        ax.plot([x_min, x_anno - 10], [y1, y1], color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.plot([x_min, x_anno - 10], [y2, y2], color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        arrow = FancyArrowPatch((x_anno, y1), (x_anno, y2), arrowstyle='<->', mutation_scale=15, color='red')
        ax.add_patch(arrow)
        text_str = f"{seg_type}: {length:.2f}\n(Σ:{cum_y[i]:.2f})"
        ax.text(x_anno - 5, mid_y, text_str, ha='right', va='center', fontsize=10, color='red',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # 7c. Summary Text
    display_thickness = thickness if thickness is not None else depth_raw
    summary_text = (f"CARDBOARD THICKNESS: {display_thickness:.4f}\n"
                    f"SCALE: {scale:.4f}\n"
                    f"CARDBOARD SIDES: {cardboard_sides if cardboard_sides else 'N/A'}")
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=14, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3, boxstyle='round,pad=0.5'))

    ax.set_title(f"Window Blueprint: {os.path.basename(obj_path)}", fontsize=18, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        print(f"Results saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path")
    parser.add_argument("--cardboard", type=float, nargs=2, help="Sides of physical cardboard (e.g., 50 60).")
    parser.add_argument("--thickness", type=float, help="Desired thickness of the cardboard (for OBJ).")
    parser.add_argument("--output", help="Path to save visualization image.")
    parser.add_argument("--output-obj", help="Path to save scaled OBJ model.")
    args = parser.parse_args()
    
    measure_and_visualize_window(args.obj_path, args.cardboard, args.thickness, args.output, args.output_obj)
