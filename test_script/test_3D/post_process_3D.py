# ./post_process_3D.py
import time, trimesh
from trimesh.smoothing import filter_laplacian
import open3d as o3d
import numpy as np


INPUT_MESH = "house_mesh.glb"
OUTPUT_MESH = "house_mesh_processed.glb"
OUT_POISSON = "house_mesh_poisson.glb"

start = time.time()

loaded = trimesh.load(INPUT_MESH, force="mesh")
if isinstance(loaded, trimesh.Scene):
    mesh = max(loaded.geometry.values(), key=lambda m: len(m.vertices))

else:
    mesh = loaded
print(f"Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")


mesh_smooth = mesh.copy()
filter_laplacian(
    mesh_smooth,
    lamb=0.2,
    iterations=50,
)
print("Applied Laplacian smoothing")

mesh_subdiv = mesh_smooth.subdivide()
print(
    f"After subdivision: {len(mesh_subdiv.vertices)} verts, {len(mesh_subdiv.faces)} faces"
)


mesh_subdiv.export(OUTPUT_MESH, file_type="glb")
print(f"Saved smoothed mesh -> {OUTPUT_MESH}")


try:
    verts = np.asarray(mesh_subdiv.vertices).copy()
    normals = np.asarray(mesh_subdiv.vertex_normals).copy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1
    )

    mesh_o3d.compute_vertex_normals()
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        vertex_normals=np.asarray(mesh_o3d.vertex_normals),
        process=False,
    )
    tm.export(OUT_POISSON, file_type="glb")
    print(f"Saved Poisson mesh -> {OUT_POISSON}")

except Exception as e:
    print("Poisson reconstruction failed:", e)
    print("Falling back to subdivided mesh only.")

print(f"Total processing time: {time.time() - start:.2f}s")
