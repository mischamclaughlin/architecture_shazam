# blender_model.py
import sys, json, random
from math import radians
import bpy, bmesh


def load_params(path):
    with open(path, "r") as f:
        return json.load(f)


def carve_window(obj, inset=0.05, depth=0.1):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    face = max(bm.faces, key=lambda f: f.normal.y)
    res = bmesh.ops.inset_region(
        bm, faces=[face], thickness=inset, depth=0  # <-- changed here
    )
    inset_face = res["faces"][0]

    extrude = bmesh.ops.extrude_face_region(bm, geom=[inset_face])
    verts = [v for v in extrude["geom"] if isinstance(v, bmesh.types.BMVert)]
    for v in verts:
        v.co.y -= depth

    bevel_edges = [e for e in bm.edges if e.select]
    bmesh.ops.bevel(bm, geom=bevel_edges, offset=inset / 2, segments=2)

    bm.to_mesh(mesh)
    bm.free()


def decorate_facade(obj, rows=3, chance=0.3, depth=0.1):
    """Subdivide the +Y face into a grid and randomly extrude panels."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    face = max(bm.faces, key=lambda f: f.normal.y)
    # collect its boundary edges
    edges = list(face.edges)
    # subdivide into roughly rows×rows grid
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=rows, use_grid_fill=True)
    # extrude a few panels
    for f in [f for f in bm.faces if f.normal.y > 0.9]:
        if random.random() < chance:
            extr = bmesh.ops.extrude_face_region(bm, geom=[f])
            verts = [v for v in extr["geom"] if isinstance(v, bmesh.types.BMVert)]
            for v in verts:
                v.co.y += random.choice([-depth, depth])

    bm.to_mesh(mesh)
    bm.free()


def generate_building(params):
    floors = int(params["floors"])
    fh = float(params["floor_height"])
    wp = int(params["windows_per_floor"])
    style = params["window_style"]
    pattern = params["facade_pattern"]
    orn = float(params["ornament_level"])
    seed = int(params.get("seed", 0))
    random.seed(seed)

    # create collection
    coll = bpy.data.collections.new("Building")
    bpy.context.scene.collection.children.link(coll)

    for i in range(floors):
        # Add cube
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, i * fh + fh / 2))
        floor = bpy.context.active_object

        # === KEY FIX: duplicate the mesh so each floor has its own data
        floor.data = floor.data.copy()

        # scale & apply transforms so BMesh sees correct dimensions
        floor.scale = (2.0, 2.0, fh / 2)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        floor.name = f"Floor_{i}"
        coll.objects.link(floor)

        # carve windows
        for _ in range(wp):
            carve_window(floor, inset=0.05 + 0.05 * orn, depth=0.1 + 0.1 * orn)

        # decorate façade
        if "staggered" in pattern:
            decorate_facade(
                floor,
                rows=3 + int(orn * 3),
                chance=0.3 + orn * 0.4,
                depth=0.1 + 0.1 * orn,
            )

    # add a simple cone roof
    bpy.ops.mesh.primitive_cone_add(
        vertices=32,
        radius1=2.2,
        depth=fh * floors * 0.2,
        location=(0, 0, floors * fh + fh * floors * 0.1),
    )
    roof = bpy.context.active_object
    roof.name = "Roof"
    coll.objects.link(roof)

    # make sure normals are consistent
    for ob in coll.objects:
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()


if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 3:
        sys.exit(
            "Usage: blender --background --python blender_model.py -- params.json output.glb"
        )
    idx = argv.index("--")
    p, out = argv[idx + 1], argv[idx + 2]
    params = load_params(p)
    generate_building(params)

    # convert everything to real mesh
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.collections["Building"].objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.collections["Building"].objects[0]
    bpy.ops.object.convert(target="MESH")

    # export GLB
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
