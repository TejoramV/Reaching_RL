import numpy as np
import cv2

extrinisic = np.array([[ 0.55741564,  0.3679865 , -0.74422694,  1.07309414],
       [ 0.82889883, -0.19585794,  0.52399084, -0.40322993],
       [ 0.0470588 , -0.90896953, -0.41419786,  0.64178561],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


intrensic = np.array([[384.55432129,   0.        , 325.71856689],
       [  0.        , 384.04806519, 243.92892456],
       [  0.        ,   0.        ,   1.        ]])


depth = cv2.imread('/home/weirdlab/data/1.png',0)
#import open3d as o3d      
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])
       
def get_pcd(depth, cam_int, cam_ext, depth_scale=1000):
       if len(depth.shape) == 3:
              depth = depth.squeeze()
       height, width = depth.shape[:2]
       depth = depth / depth_scale
       xlin = np.linspace(0, width - 1, width)
       ylin = np.linspace(0, height - 1, height)
       px, py = np.meshgrid(xlin, ylin)
       px = (px - cam_int[0, 2]) * (depth / cam_int[0, 0])
       py = (py - cam_int[1, 2]) * (depth / cam_int[1, 1])
       points = np.stack((px, py, depth, np.ones(depth.shape)), axis=-1)
       points = (cam_ext @ points.reshape(-1, 4).T).T
       points = points[:, :3]
       return points
print(get_pcd(depth,intrensic,extrinisic,1000).shape)
x=get_pcd(depth,intrensic,extrinisic,1000)[:,0]
y=get_pcd(depth,intrensic,extrinisic,1000)[:,1]
z=get_pcd(depth,intrensic,extrinisic,1000)[:,2]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

# Generate example point cloud data (replace this with your own data)


# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the points
ax.scatter(x, y, z, c='b', marker='.')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()
