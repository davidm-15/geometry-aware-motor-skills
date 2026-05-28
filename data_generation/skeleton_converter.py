import trimesh
import numpy as np
import json
import os
import argparse
from pathlib import Path

class WindowSkeletonConverter:
    """
    Converts window CAD models into a simplified skeletal representation
    consisting of nodes and edges.
    """
    
    def __init__(self, alignment_tolerance=0.1, cluster_tolerance=2.0, min_span_ratio=0.1):
        self.alignment_tolerance = alignment_tolerance
        self.cluster_tolerance = cluster_tolerance
        self.min_span_ratio = min_span_ratio

    def extract_skeleton_from_file(self, obj_path):
        """
        Loads an OBJ file and extracts the skeleton as a graph.
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"File {obj_path} does not exist.")
            
        mesh = trimesh.load(obj_path, force='mesh')
        skeleton = self.extract_skeleton(mesh)
        
        # Update metadata to include the specific file path
        skeleton["metadata"]["source_obj"] = os.path.abspath(obj_path)
        return skeleton

    def extract_skeleton(self, mesh):
        """
        Parses the trimesh object and extracts the skeleton as a graph.
        """
        # 1. Determine orientation (smallest dimension is depth)
        extents = mesh.extents
        bounds = mesh.bounds
        bounds_center = (bounds[0] + bounds[1]) / 2
        axis_depth = np.argmin(extents)
        axes_2d = [i for i in range(3) if i != axis_depth]
        axis_x, axis_y = axes_2d[0], axes_2d[1]
        
        # 2. Extract 2D Profile (cross-section at center)
        normal = np.zeros(3)
        normal[axis_depth] = 1.0
        slice_3d = mesh.section(plane_origin=bounds_center, plane_normal=normal)
        
        if slice_3d is None:
            raise ValueError("Could not extract cross-section from mesh.")
            
        # 3. Identify structural levels (horizontal and vertical bars)
        levels_x = self._find_levels(slice_3d, axis_x, axis_y, extents[axis_y])
        levels_y = self._find_levels(slice_3d, axis_y, axis_x, extents[axis_x])
        
        if len(levels_x) < 2 or len(levels_y) < 2:
            # Fallback to outer bounds if no internal structure is detected
            levels_x = [bounds[0][axis_x], bounds[1][axis_x]]
            levels_y = [bounds[0][axis_y], bounds[1][axis_y]]
            
        # 4. Filter levels to get "bar centers"
        grid_x = self._cluster_to_centers(levels_x)
        grid_y = self._cluster_to_centers(levels_y)
        
        # 5. Build Graph Nodes and Edges
        nodes = []
        node_map = {} # (idx_x, idx_y) -> node_index
        
        # All intersections in the grid
        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                node_map[(i, j)] = len(nodes)
                # Map back to 3D space (filling axis_depth with 0 or center)
                pos = [0.0, 0.0, 0.0]
                pos[axis_x] = float(x)
                pos[axis_y] = float(y)
                pos[axis_depth] = float(bounds_center[axis_depth])
                nodes.append(pos)
                
        edges = []
        # Build edges based on connectivity in the slice
        # Vertical segments
        for i in range(len(grid_x)):
            for j in range(len(grid_y) - 1):
                if self._is_segment_present(slice_3d, grid_x[i], grid_y[j], grid_x[i], grid_y[j+1], axis_x, axis_y):
                    edges.append([node_map[(i, j)], node_map[(i, j+1)]])
                    
        # Horizontal segments
        for j in range(len(grid_y)):
            for i in range(len(grid_x) - 1):
                if self._is_segment_present(slice_3d, grid_x[i], grid_y[j], grid_x[i+1], grid_y[j], axis_x, axis_y):
                    edges.append([node_map[(i, j)], node_map[(i+1, j)]])
                    
        # 6. Simplify Graph (Collapse nodes within bar thickness)
        final_nodes, final_edges = self._simplify_graph(nodes, edges, threshold=100.0)
        
        # 7. Compute Node Degrees
        degrees = [0] * len(final_nodes)
        for edge in final_edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
                    
        return {
            "nodes": final_nodes,
            "edges": final_edges,
            "degrees": degrees,
            "metadata": {
                "source_obj": "Direct mesh input", # Can be overwritten if called via extract_skeleton_from_file
                "extents": extents.tolist(),
                "max_extent": float(np.max(extents)),
                "axis_depth": int(axis_depth)
            }
        }

    def _simplify_graph(self, nodes, edges, threshold=100.0):
        """
        Merges nodes that are closer than threshold and simplifies edges.
        This collapses the 2x2 or 2x1 grid points at crossings into single points.
        """
        if not nodes:
            return [], []
            
        nodes = np.array(nodes)
        num_nodes = len(nodes)
        
        # 1. Cluster nodes
        # Simple greedy clustering: for each node, find all nodes within threshold
        node_to_cluster = np.arange(num_nodes)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(nodes[i] - nodes[j]) < threshold:
                    # Merge cluster j into cluster i
                    cluster_j = node_to_cluster[j]
                    cluster_i = node_to_cluster[i]
                    node_to_cluster[node_to_cluster == cluster_j] = cluster_i
                    
        # 2. Recompute new nodes (centroids of clusters)
        unique_clusters = np.unique(node_to_cluster)
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        
        final_nodes = []
        for cluster_id in unique_clusters:
            cluster_pts = nodes[node_to_cluster == cluster_id]
            final_nodes.append(np.mean(cluster_pts, axis=0).tolist())
            
        # 3. Recompute edges
        final_edges_set = set()
        for edge in edges:
            n1 = cluster_map[node_to_cluster[edge[0]]]
            n2 = cluster_map[node_to_cluster[edge[1]]]
            if n1 != n2:
                # Store as sorted tuple to avoid duplicates like (1,2) and (2,1)
                final_edges_set.add(tuple(sorted((n1, n2))))
                
        final_edges = [list(e) for e in sorted(list(final_edges_set))]
        
        return final_nodes, final_edges

    def _find_levels(self, slice_path, axis_idx, other_axis_idx, other_span):
        """Identify significant axis-aligned levels."""
        votes = []
        for entity in slice_path.entities:
            discrete = entity.discrete(slice_path.vertices)
            for i in range(len(discrete)-1):
                p1, p2 = discrete[i], discrete[i+1]
                if abs(p1[axis_idx] - p2[axis_idx]) < self.alignment_tolerance:
                    length = abs(p1[other_axis_idx] - p2[other_axis_idx])
                    votes.append((p1[axis_idx], length))
        
        if not votes: return []
        
        votes.sort()
        clusters = []
        curr_coord, curr_weight = votes[0]
        for coord, weight in votes[1:]:
            if coord - curr_coord < self.cluster_tolerance:
                curr_coord = (curr_coord * curr_weight + coord * weight) / (curr_weight + weight)
                curr_weight += weight
            else:
                clusters.append((curr_coord, curr_weight))
                curr_coord, curr_weight = coord, weight
        clusters.append((curr_coord, curr_weight))
        
        # Filter for major levels
        significant = [c[0] for c in clusters if c[1] > other_span * self.min_span_ratio]
        return sorted(significant)

    def _cluster_to_centers(self, levels):
        """
        Groups boundary pairs into single center lines.
        Assuming windows have bars with thickness, boundary detection finds both edges.
        We merge pairs into their midpoints.
        """
        if len(levels) < 2: return levels
        
        # Sort levels just in case
        levels = sorted(levels)
        centers = []
        i = 0
        while i < len(levels):
            # If two levels are close (a bar's two edges), merge them.
            # Window frames are usually 30-70mm thick. 
            # If we see a gap < 40mm, it's likely a single bar pair.
            if i + 1 < len(levels) and (levels[i+1] - levels[i]) < 40.0:
                centers.append((levels[i] + levels[i+1]) / 2.0)
                i += 2
            else:
                centers.append(levels[i])
                i += 1
        return centers

    def _is_segment_present(self, slice_path, x1, y1, x2, y2, axis_x, axis_y, tolerance=50.0):
        """
        Checks if there's actually geometry connecting these two grid points.
        """
        # Check midpoint and quarter-points to be more robust
        test_points = [
            (0.5, 0.5), # Midpoint
            (0.25, 0.25),
            (0.75, 0.75)
        ]
        
        for tx, ty in test_points:
            mx = x1 + (x2 - x1) * tx
            my = y1 + (y2 - y1) * ty
            
            mid_p = np.array([0.0, 0.0, 0.0])
            mid_p[axis_x] = mx
            mid_p[axis_y] = my
            
            found_near = False
            for entity in slice_path.entities:
                discrete = entity.discrete(slice_path.vertices)
                for i in range(len(discrete)-1):
                    p1, p2 = discrete[i], discrete[i+1]
                    v = p2 - p1
                    w = mid_p - p1
                    l2 = np.sum(v**2)
                    if l2 == 0:
                        d = np.linalg.norm(mid_p - p1)
                    else:
                        t = max(0, min(1, np.dot(w, v) / l2))
                        projection = p1 + t * v
                        d = np.linalg.norm(mid_p - projection)
                    
                    if d < tolerance:
                        found_near = True
                        break
                if found_near: break
            
            if found_near:
                return True # If any test point is near geometry, we assume the segment exists
                
        return False

    def visualize_skeleton(self, skeleton, output_png):
        """
        Generates a 2D PNG visualization of the skeleton.
        """
        import matplotlib.pyplot as plt
        
        nodes = np.array(skeleton["nodes"])
        edges = skeleton["edges"]
        metadata = skeleton["metadata"]
        axis_depth = metadata["axis_depth"]
        axes_2d = [i for i in range(3) if i != axis_depth]
        ax_x, ax_y = axes_2d[0], axes_2d[1]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Plot edges
        for edge in edges:
            p1, p2 = nodes[edge[0]], nodes[edge[1]]
            ax.plot([p1[ax_x], p2[ax_x]], [p1[ax_y], p2[ax_y]], 'b-', linewidth=2)
            
        # Plot nodes
        ax.scatter(nodes[:, ax_x], nodes[:, ax_y], c='r', s=50, zorder=5)
        
        ax.set_title(f"Skeleton: {os.path.basename(metadata['source_obj'])}")
        ax.set_xlabel(f"Axis {ax_x} (mm)")
        ax.set_ylabel(f"Axis {ax_y} (mm)")
        
        plt.savefig(output_png, bbox_inches='tight', dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Convert window OBJs to skeletons.")
    parser.add_argument("input", help="Input .obj file or directory.")
    parser.add_argument("--output", help="Output directory for skeletons (default: datasets/windows-v2/0_skeletons).")
    parser.add_argument("--no-vis", action="store_true", help="Disable PNG visualization.")
    args = parser.parse_args()
    
    default_output = Path("datasets/windows-v2/0_skeletons")
    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = WindowSkeletonConverter()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        skeleton = converter.extract_skeleton_from_file(str(input_path))
        sample_name = input_path.stem
        
        # Save JSON
        json_file = output_dir / f"{sample_name}.json"
        with open(json_file, 'w') as f:
            json.dump(skeleton, f, indent=2)
        print(f"Skeleton JSON saved to {json_file}")
        
        # Save PNG
        if not args.no_vis:
            png_file = output_dir / f"{sample_name}.png"
            converter.visualize_skeleton(skeleton, str(png_file))
            print(f"Skeleton PNG saved to {png_file}")
            
    elif input_path.is_dir():
        # Batch process directory
        obj_files = list(input_path.glob("**/[0-9]*_wr1fr_1.obj"))
        print(f"Found {len(obj_files)} OBJ files. Outputting to {output_dir}")
        
        for obj_file in obj_files:
            try:
                skeleton = converter.extract_skeleton(str(obj_file))
                sample_name = obj_file.parent.name # Use folder name as ID (e.g. 1_wr1fr_1)
                
                # Save JSON
                json_file = output_dir / f"{sample_name}.json"
                with open(json_file, 'w') as f:
                    json.dump(skeleton, f, indent=2)
                
                # Save PNG
                if not args.no_vis:
                    png_file = output_dir / f"{sample_name}.png"
                    converter.visualize_skeleton(skeleton, str(png_file))
                    
            except Exception as e:
                print(f"Failed to process {obj_file}: {e}")
        
        print(f"Batch processing complete. Results in {output_dir}")

if __name__ == "__main__":
    main()
