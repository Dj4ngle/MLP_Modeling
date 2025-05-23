import open3d as o3d
import numpy as np
import random
import matplotlib.pyplot as plt


# ==== Метрика качества ====
def score(triangles, holes):
    return triangles - holes * 2.0


# ==== Оценка меша ====
def evaluate_mesh(mesh):
    num_triangles = len(mesh.triangles)

    # Вручную ищем граничные рёбра
    edges = dict()
    for tri in np.asarray(mesh.triangles):
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edges[edge] = edges.get(edge, 0) + 1
    boundary_edges = [e for e, c in edges.items() if c == 1]
    num_holes = len(boundary_edges)

    return num_triangles, num_holes


# ==== Основной алгоритм ====
def optimize_radii_hill_climb(pcd, steps=50):
    n_points = len(pcd.points)
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 2.0, max_nn=30)
    )

    # Инициализация радиусов
    best_radii = [0.5 * avg_dist, 1.0 * avg_dist, 1.5 * avg_dist]
    best_score = float('-inf')
    best_mesh = None
    history = []

    for step in range(steps):
        # Мутация радиусов
        new_radii = [max(0.001, r + random.uniform(-0.05, 0.05) * avg_dist) for r in best_radii]

        # BPA
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(new_radii)
            )
        except Exception as e:
            print(f"Step {step+1:02d}: BPA failed: {e}")
            continue

        triangles, holes = evaluate_mesh(mesh)
        current_score = score(triangles, holes)
        history.append(current_score)

        print(f"Step {step+1:02d}: radii={np.round(new_radii, 5)}, triangles={triangles}, holes={holes}, score={current_score}")

        if current_score > best_score:
            best_radii = new_radii
            best_score = current_score
            best_mesh = mesh

    return best_radii, best_score, best_mesh, history


# ==== Точка входа ====
if __name__ == "__main__":
    # Замените путь на свой файл
    pcd = o3d.io.read_point_cloud("clouds/export1_good1.pcd")

    best_radii, best_score, best_mesh, score_history = optimize_radii_hill_climb(pcd)

    print("\n🎯 Final Best Radii:", best_radii)
    print("✅ Final Score:", best_score)

    # Визуализация финального результата
    best_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([best_mesh], window_name="Final BPA Mesh")

    # 💾 Сохранение в GLB
    o3d.io.write_triangle_mesh("output_mesh.glb", best_mesh, write_triangle_uvs=False)
    print("💾 Модель сохранена как output_mesh.glb")

    # График
    plt.plot(score_history)
    plt.title("Hill Climbing Optimization")
    plt.xlabel("Step")
    plt.ylabel("Score (triangles - 10 * holes)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
