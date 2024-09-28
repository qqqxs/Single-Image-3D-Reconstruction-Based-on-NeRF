import numpy as np
import open3d as o3d

# 加载点云数据
vertices_cano = np.load('vertices_cano.npy')
vertices_color_cano = np.load('vertices_color_cano.npy')
# vertices_novel = np.load('vertices_novel.npy')
# vertices_color_novel = np.load('vertices_color_novel.npy')

# 将顶点和颜色数据合并
# all_v = np.concatenate((vertices_cano, vertices_novel), axis=0)
# all_v_color = np.concatenate((vertices_color_cano, vertices_color_novel), axis=0)

# 创建 open3d 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices_cano)
pcd.colors = o3d.utility.Vector3dVector(vertices_color_cano)

# 可视化点云
o3d.visualization.draw_geometries([pcd])
