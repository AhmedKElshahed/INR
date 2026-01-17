import open3d as o3d
import trimesh
import numpy as np
import os
import gc
import argparse # <--- NEW IMPORT

def generate_data(mesh_path, output_path, num_samples=500000):
    print(f"--- PROCESSING {mesh_path} ---")
    print(f"--- OUTPUT TARGET {output_path} ---")
    
    # 1. Load and Simplify
    print("-> Loading mesh with Open3D...")
    try:
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    if len(mesh_o3d.vertices) == 0:
        print("Error: Mesh is empty.")
        return

    print(f"   Original faces: {len(mesh_o3d.triangles)}")
    
    # Simplify to 50k faces
    target_faces = 50000
    if len(mesh_o3d.triangles) > target_faces:
        print(f"-> Simplifying to {target_faces} faces...")
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        
    # 2. Convert to Trimesh
    print("-> Converting to Trimesh...")
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    del mesh_o3d
    gc.collect()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 3. Normalize
    print("-> Normalizing coordinates...")
    vertices = mesh.vertices
    center = (vertices.max(0) + vertices.min(0)) / 2
    scale = 1.8 / (vertices.max(0) - vertices.min(0)).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    # 4. Sample Points
    print(f"-> Sampling {num_samples} points...")
    n_surf = num_samples // 2
    n_uni = num_samples - n_surf
    p_uni = np.random.rand(n_uni, 3) * 2 - 1
    p_surf, _ = trimesh.sample.sample_surface(mesh, n_surf)
    p_surf += np.random.normal(0, 0.002, p_surf.shape)
    all_points = np.concatenate([p_uni, p_surf], axis=0)
    
    # 5. Calculate Occupancy
    print("-> Ray tracing occupancy...")
    occupancy = np.zeros(len(all_points), dtype=np.uint8) 
    chunk_size = 2000 
    for i in range(0, len(all_points), chunk_size):
        end = min(i + chunk_size, len(all_points))
        batch_points = all_points[i:end]
        occupancy[i:end] = mesh.contains(batch_points)
        if i % (chunk_size * 50) == 0:
            print(f"   Checked {end}/{len(all_points)} points...")

    # 6. Save
    print(f"-> Saving to {output_path}...")
    np.savez_compressed(output_path, points=all_points.astype(np.float32), occupancies=occupancy)
    print("DONE. Data ready for training.")

if __name__ == "__main__":
    # ARGUMENT PARSING LOGIC
    parser = argparse.ArgumentParser(description="Generate 3D occupancy data from a mesh.")
    parser.add_argument("--mesh", type=str, default="dragon.obj", help="Path to the input .obj file")
    parser.add_argument("--samples", type=int, default=500000, help="Number of points to sample")
    args = parser.parse_args()

    if os.path.exists(args.mesh):
        # Create output name automatically: "spot.obj" -> "spot_dataset.npz"
        base_name = os.path.splitext(args.mesh)[0]
        output_filename = f"{base_name}_dataset.npz"
        
        generate_data(args.mesh, output_filename, num_samples=args.samples)
    else:
        print(f"Error: File '{args.mesh}' not found!")
