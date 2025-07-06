# ./visualise.py
import trimesh

mesh = trimesh.load("outputs/meshes/coarse_model.obj")
mesh.show()
