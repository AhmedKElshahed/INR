import open3d as o3d
import trimesh
import numpy as np
import os
import gc

def generate_data(mesh_path, output_path="dragon_dataset.npz", num_samples=500000):
    print(f"--- PROCESSING {mesh_path} ---")
    
    # 1. Load and Simplify using Open3D (Fast & Stable)
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
    
    # Simplify if too heavy (keeps shape, reduces RAM usage)
    if len(mesh_o3d.triangles) > 10000:
        print("-> Simplifying to 10k faces...")
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=10000)
        
    # 2. Convert to Trimesh for Ray Tracing
    print("-> Converting to Trimesh...")
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    
    # Open3D objects are heavy, delete immediately
    del mesh_o3d
    gc.collect()
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 3. Normalize to Unit Sphere [-1, 1]
    print("-> Normalizing coordinates...")
    vertices = mesh.vertices
    center = (vertices.max(0) + vertices.min(0)) / 2
    scale = 1.8 / (vertices.max(0) - vertices.min(0)).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    # 4. Sample Points
    print(f"-> Sampling {num_samples} points...")
    # Split: 50% near surface (details), 50% random uniform (empty space)
    n_surf = num_samples // 2
    n_uni = num_samples - n_surf
    
    # Uniform sampling
    p_uni = np.random.rand(n_uni, 3) * 2 - 1
    
    # Surface sampling with slight noise
    p_surf, _ = trimesh.sample.sample_surface(mesh, n_surf)
    p_surf += np.random.normal(0, 0.01, p_surf.shape)
    
    all_points = np.concatenate([p_uni, p_surf], axis=0)
    
    # 5. Calculate Occupancy (The slow part)
    print("-> Ray tracing occupancy (this takes time)...")
    
    # We process in chunks to prevent RAM spikes
    occupancy = np.zeros(len(all_points), dtype=np.uint8) 
    chunk_size = 100000
    
    for i in range(0, len(all_points), chunk_size):
        end = min(i + chunk_size, len(all_points))
        batch_points = all_points[i:end]
        # .contains returns bool, we save as uint8 (0 or 1)
        occupancy[i:end] = mesh.contains(batch_points)
        print(f"   Checked {end}/{len(all_points)} points...")

    # 6. Save to Disk
    print(f"-> Saving to {output_path}...")
    np.savez_compressed(output_path, points=all_points.astype(np.float32), occupancies=occupancy)
    print("DONE. Data ready for training.")

if __name__ == "__main__":
    # Install open3d if missing: pip install open3d
    if os.path.exists("dragon.obj"):
        generate_data("dragon.obj", "dragon_dataset.npz", num_samples=500000)
    else:
        print("Please download dragon.obj first!")
