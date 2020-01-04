import pymesh
import numpy as np
import matplotlib.pyplot as plt
import os 

def compare_curvature(gaussian_, mean_):
    cmmon_path = os.path.commonpath([gaussian_, mean_])
    print(cmmon_path)
    gaussian_mesh = pymesh.load_mesh(gaussian_)
    mean_mesh = pymesh.load_mesh(mean_)

    print(gaussian_mesh.get_attribute_names())
    attrs = ['vertex_red', 'vertex_green', 'vertex_blue']
    for attr_name in attrs:
        diff = abs(gaussian_mesh.get_vertex_attribute(attr_name) - \
                    mean_mesh.get_vertex_attribute(attr_name))
        gaussian_mesh.set_attribute(attr_name, diff)

    pymesh.save_mesh(os.path.join(cmmon_path, "gaussian_mean_taubian_diff.ply"), gaussian_mesh, *attrs, ascii=True)


def edge_length(edge):
    return np.sqrt(sum([x**2 for x in edge]))


def calculate_mesh_quality(mesh):
    mesh.enable_connectivity()
    mesh.add_attribute('face_area')
    faces_areas = mesh.get_face_attribute('face_area')
    qualities, skewness = [], []
    print(mesh.num_faces)
    equi_angle = np.pi / 3
    ang_180 = np.pi / 2
    for i, face in enumerate(mesh.faces):
        # face
        q = 4 * np.sqrt(3) * faces_areas[i]

        v1, v2, v3 = mesh.vertices[face]
        e1 = v2 - v1
        e2 = v3 - v1
        e3 = v3 - v2

        edges = [e1, e2, e3]

        edges_lengths = [edge_length(edge) for edge in edges]

        a1 = np.arccos(np.dot(e1, e2) / (edges_lengths[0] * edges_lengths[1]))
        a2 = np.arccos(np.dot(e1, e3) / (edges_lengths[0] * edges_lengths[2]))
        a3 = np.arccos(np.dot(e3, e2) / (edges_lengths[2] * edges_lengths[1]))

        angles = [a1, a2, a3]

        angles = [a if (a < ang_180) else (np.pi - a) for a in angles]
        print(angles)
        max_angle = max(angles)
        min_angle = min(angles)

        sk = np.max([(max_angle - equi_angle) / (np.pi - equi_angle),
                     (equi_angle - min_angle) / equi_angle])

        edges_len = sum([e**2 for e in edges_lengths])
        q /= edges_len
        qualities.append(q)
        skewness.append(sk)

    skewness = np.array(skewness)
    qualities = np.array(qualities)
    np.save('qualities.npy', qualities)
    np.save('skewness.npy', skewness)


def edge_skewness(edges):
    angle_collection = []
    for e, edge in edges:
        reduced_edges = [edges[k] for k in range(3) if k != e]
        mid_e1 = reduced_edges[0] / 2
        mid_e2 = reduced_edges[1] / 2

        a1 = np.arccos(
            np.dot(mid_e1, mid_e2) / edge_length(mid_e2) * edge_length(mid_e1))
        a2 = np.pi - a1

        angle_collection.append(a1, a2)
    return angle_collection


# mesh = pymesh.load_mesh('sphere.ply')
# calculate_mesh_quality(mesh)
gmesh = 'meshes/bunny/reconstruction/bun_zipper_gauss_taubin_apr.ply'
mmesh = 'meshes/bunny/reconstruction/bun_zipper_mean_taubin_apr.ply'
compare_curvature(gmesh, mmesh)
