import open3d as o3d
import numpy as np
import random
import os
import csv


def evaluate_mesh(mesh):
    num_triangles = len(mesh.triangles)
    edges = dict()
    for tri in np.asarray(mesh.triangles):
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edges[edge] = edges.get(edge, 0) + 1
    boundary_edges = [e for e, c in edges.items() if c == 1]
    num_holes = len(boundary_edges)
    return num_triangles, num_holes


def score(triangles, holes):
    return triangles - holes * 2.0


def optimize_radii(pcd, steps=30):
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 2.0, max_nn=30)
    )

    best_radii = [0.5 * avg_dist, 1.0 * avg_dist, 1.5 * avg_dist]
    best_score = float('-inf')

    for _ in range(steps):
        new_radii = [max(0.001, r + random.uniform(-0.05, 0.05) * avg_dist) for r in best_radii]
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(new_radii)
            )
        except:
            continue

        triangles, holes = evaluate_mesh(mesh)
        s = score(triangles, holes)

        if s > best_score:
            best_radii = new_radii
            best_score = s

    return best_radii


def build_dataset_from_folder(folder_path, output_csv="bpa_dataset.csv"):
    rows = []
    files = [f for f in os.listdir(folder_path) if f.endswith((".ply", ".pcd", ".xyz"))]

    for filename in files:
        path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) < 100:  # фильтрация мусора
            continue

        num_points = len(pcd.points)
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        r1, r2, r3 = optimize_radii(pcd)
        rows.append([num_points, avg_dist, r1, r2, r3])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_points", "avg_dist", "r1", "r2", "r3"])
        writer.writerows(rows)

    print(f"\n✅ Dataset saved to {output_csv}")


if __name__ == "__main__":
    build_dataset_from_folder("clouds")
