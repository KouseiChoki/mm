import open3d as o3d
import sys
# 读取 PLY 文件
ply_file = sys.argv[1]
point_cloud = o3d.io.read_point_cloud(ply_file)

# 打印点云信息
print(point_cloud)
print(f"Point cloud has {len(point_cloud.points)} points.")

# 创建一个坐标轴对象
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5,  # 坐标轴的长度
    origin=[0, 0, 0]  # 坐标轴的原点
)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud,coordinate_frame],
                                  window_name="PLY Point Cloud Viewer",
                                  width=1900,
                                  height=1080,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)