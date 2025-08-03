# ./visualise.py
import trimesh

mesh = trimesh.load("generated_3D_models/house_mesh_0_20250803_152258.obj")
mesh.show()
