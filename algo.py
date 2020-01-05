import pymesh
import numpy as np
# import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import os
import itertools


def mesh_op(filename):
    mesh = pymesh.load_mesh(filename)
    assembler = pymesh.Assembler(mesh)
    L = assembler.assemble("laplacian")
    print(L.shape, mesh.num_vertices, mesh.num_faces)
    eigenvals, eigenvecs = eigs(L, k=1000)
    print(eigenvecs.shape, eigenvals.shape)
    np.save("eigenvals.npy", eigenvals)
    np.save("eigenvecs.npy", eigenvecs)


def compare_curvature(gaussian_, mean_, save_npy=True):
    cmmon_path = os.path.commonpath([gaussian_, mean_])
    # print(cmmon_path, gaussian_, mean_)
    gaussian_mesh = pymesh.load_mesh(gaussian_)
    mean_mesh = pymesh.load_mesh(mean_)

    # print(gaussian_mesh.get_attribute_names())
    attrs = ['vertex_red', 'vertex_green', 'vertex_blue']
    color_npy = np.zeros(shape=(mean_mesh.num_vertices, 3), dtype=np.int32)
    g_colors = np.zeros(shape=(mean_mesh.num_vertices, 3), dtype=np.int32)
    m_colors = np.zeros(shape=(mean_mesh.num_vertices, 3), dtype=np.int32)

    for i, attr_name in enumerate(attrs):
        diff = abs(gaussian_mesh.get_vertex_attribute(attr_name) - \
                    mean_mesh.get_vertex_attribute(attr_name))
        gaussian_mesh.set_attribute(attr_name, diff)
        color_npy[:, i] = diff.reshape(-1, )
        m_colors[:, i] = mean_mesh.get_vertex_attribute(attr_name).reshape(
            -1, )
        g_colors[:, i] = gaussian_mesh.get_vertex_attribute(attr_name).reshape(
            -1, )

    splits = os.path.basename(gaussian_.replace('.ply', '')).split('_')
    base_file, curv1, method1 = '_'.join(splits[:2]), splits[-2], splits[-1]

    splits = os.path.basename(mean_.replace('.ply', '')).split('_')
    base_file, curv2, method2 = '_'.join(splits[:2]), splits[-2], splits[-1]
    if save_npy:
        root_path = 'meshes/mesh_data/curvature_comp'

        svp = os.path.join(
            root_path,
            f"diff/{base_file}_{curv1}_{curv2}_{method1}_{method2}_diff_colors.npy"
        )

        print(f"c1 {curv1} c2 {curv2}, m1 {method1}, m2 {method2}")
        # print(f"Saving colors to {svp}")
        np.save(svp, color_npy)
        svp = os.path.join(root_path,
                           f"base/{base_file}_{curv1}_{method1}_colors.npy")
        np.save(svp, g_colors)
        svp = os.path.join(root_path,
                           f"base/{base_file}_{curv2}_{method2}_colors.npy")
        np.save(svp, m_colors)

    # pymesh.save_mesh(os.path.join(
    #     root_path, f"mesh/{base_file}_{curv1}_{method1}_{curv1}_{method2}.ply"),
    #                  gaussian_mesh,
    #                  *attrs,
    #                  ascii=True)


def edge_length(edge):
    return np.sqrt(sum([x**2 for x in edge]))


def calculate_mesh_quality(meshname):
    mesh = pymesh.load_mesh(meshname)

    mesh.enable_connectivity()
    mesh.add_attribute('face_area')
    faces_areas = mesh.get_face_attribute('face_area')
    qualities, skewness, min_angles = [], [], []
    print(f"Mesh contains {mesh.num_faces} faces")

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
        # print(angles)
        max_angle = max(angles)
        min_angle = min(angles)

        sk = np.max([(max_angle - equi_angle) / (np.pi - equi_angle),
                     (equi_angle - min_angle) / equi_angle])

        edges_len = sum([e**2 for e in edges_lengths])
        q /= edges_len
        qualities.append(q)
        skewness.append(sk)
        min_angles.append(min_angle)

    skewness = np.array(skewness)
    qualities = np.array(qualities)
    basename = os.path.basename(meshname).replace('.ply', '')
    np.save(f'meshes/mesh_data/quality_data/{basename}_qualities.npy',
            qualities)
    np.save(f'meshes/mesh_data/quality_data/{basename}_skewness.npy', skewness)
    np.save(f'meshes/mesh_data/quality_data/{basename}_areas.npy', faces_areas)
    np.save(f'meshes/mesh_data/quality_data/{basename}_min_angles.npy',
            min_angles)


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


def aggregate_mesh_quality_data():
    for mesh_filename in [
            'meshes/bunny/reconstruction/bun_zipper.ply',
            'meshes/skeleton/hand.ply', 'meshes/sphere/sphere.ply'
    ]:
        print(f"Calculating qualities for {mesh_filename}...")
        calculate_mesh_quality(mesh_filename)


def aggregate_cmp_data():
    methods = ['normal', 'principal', 'pseudoinverse', 'taubin_apr']
    curvatures = ['gauss', 'mean']

    curv_dir = 'meshes/bunny/reconstruction/curvatures'
    fns = [
        os.path.join(curv_dir, fn) for fn in os.listdir(curv_dir)
    ]

    for element in itertools.combinations(fns, r=2):
        if element[0] != element[1]:
            # print(f"Processing {element}")
            compare_curvature(*element, save_npy=True)


aggregate_cmp_data()
