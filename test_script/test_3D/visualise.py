# ./test_script/test_3D/visualise.py
import trimesh

mesh = trimesh.load("generated_3D_models/mesh_0_20250803_180942.obj")
mesh.show()
