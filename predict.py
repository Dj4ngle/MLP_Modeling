import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import argparse


# ==== Модель (должна соответствовать train_model.py) ====
class RadiusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# ==== Загрузка модели ====
def load_model(path="model.pt"):
    checkpoint = torch.load(path, weights_only=False)
    model = RadiusNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scaler_mean"], checkpoint["scaler_scale"]


# ==== Предсказание радиусов ====
def predict_radii(model, mean, scale, num_points, avg_dist):
    x = np.array([[num_points, avg_dist]])
    x_scaled = (x - mean) / scale
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        r_pred = model(x_tensor)[0].numpy()
    scaled_r = [max(0.01, r * avg_dist) for r in r_pred]  # масштабируем
    return scaled_r


# ==== Визуализация BPA ====
def reconstruct_and_visualize(pcd, radii):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=np.mean(radii) * 2.0, max_nn=30)
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.compute_vertex_normals()
    print(f"🎯 Predicted Radii: {np.round(radii, 5)}")
    o3d.visualization.draw_geometries([mesh], window_name="Predicted BPA Mesh")


# ==== Основной запуск ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to point cloud file (.ply, .pcd, .xyz)")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) < 100:
        print("⛔ Too few points in cloud.")
        exit()

    num_points = len(pcd.points)
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    print(f"ℹ️ num_points={num_points}, avg_dist={avg_dist:.6f}")

    model, mean, scale = load_model("model.pt")
    radii = predict_radii(model, mean, scale, num_points, avg_dist)

    reconstruct_and_visualize(pcd, radii)
