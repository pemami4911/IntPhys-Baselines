import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Depth2BEV:
  """
  Implements methods to convert depth map 
  data from RGBD to a birds eye view (BEV)
  map with multiple height channels.

  Also parses the annotations file to obtain object
  labels.

  Note:
  point_cloud[:,:,0] = x (forward)
  point_cloud[:,:,1] = y (right)
  point_cloud[:,:,2] = z (up)
  
  We used a fixed max depth (2500) and rescale each scene to this depth.
  This is to enable us to use the same fixed depth for the test set, 
  where we do not know the actual max depth as status.json is not
  available on the test set.

  BEV grid map: 348 x 250 x 35. This is W x H x C.
  Equivalenty, x=348, y=250, z=35
  In PyTorch, this is 35 x 250 x 348 (NCHW format).

  """
  def __init__(self, image_dims=[288, 288], fovx=90, fovy=90, 
      bev_x_res=10, bev_y_res=10, z_channels=35,
      bev_dims=[3480, 2500, 800], fixed_depth=2500):
    self.frame_height = image_dims[0]
    self.frame_width = image_dims[1]
    self.camera_fov_x = fovx
    self.camera_fov_y = fovy
    # focal length x
    self.fx = 1 / ((2 / self.frame_width) * np.tan((self.camera_fov_x / 2) * np.pi / 180))
    # focal length y
    self.fy = 1 / ((2 / self.frame_height) * np.tan((self.camera_fov_y / 2) * np.pi / 180))
    self.cx = self.frame_width / 2
    self.cy = self.frame_height / 2
    self.bev_x_res = bev_x_res
    self.bev_y_res = bev_y_res
    self.bev_z_channels = z_channels
    self.bev_x_dim = bev_dims[0]
    self.bev_y_dim = bev_dims[1]
    self.bev_z_dim = bev_dims[2]
    self.fixed_depth = fixed_depth

  def depth_2_point_cloud(self, depth_data, max_depth=None):
    if not max_depth:
      max_depth = self.fixed_depth
    depth_data = (depth_data * max_depth) / np.max(np.abs(depth_data))
    point_cloud = np.zeros((self.frame_width, self.frame_height, 3), dtype=np.float32)
    for r in range(self.frame_height):
      for c in range(self.frame_width):
        if depth_data[r][c] > 0.0:
          v = -(r - self.cy)
          u = (c - self.cx)
          Z = depth_data[r][c]
          # Unreal engine uses left-hand XZY coordinates with z-up
          point_cloud[r, c, :] = [Z, u*Z/self.fx, v*Z/self.fy]
    return point_cloud

  def display_point_cloud(self, point_cloud, view='3d', objects=None):
    if view=='3d':
      # flattened pc for plotting
      point_cloud_flt = np.reshape(point_cloud, (point_cloud.shape[0] * point_cloud.shape[1], 3))
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111, projection='3d')  
      ax.scatter(point_cloud[:, :, 0], point_cloud[:, :, 1], point_cloud[:, :, 2], s=0.75, c=point_cloud_flt[:, 2], cmap='gray')
      ax.set_xlim(np.min(point_cloud[:,:,0]), np.max(point_cloud[:,:,0]))
      ax.set_ylim(np.min(point_cloud[:,:,1]), np.max(point_cloud[:,:,1]))
      ax.set_zlim(np.min(point_cloud[:,:,2]), np.max(point_cloud[:,:,2]))
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")
    elif view == 'x-y':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 1], point_cloud[:, :, 0], s=0.75, cmap='gray')
      ax.set_xlim(np.min(point_cloud[:,:,1]), np.max(point_cloud[:,:,1]))  # y is <- ->
      ax.set_ylim(np.min(point_cloud[:,:,0]), np.max(point_cloud[:,:,0]))  # x is depth
    elif view == 'y-z':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 1], point_cloud[:, :, 2], s=0.75, cmap='gray')
      ax.set_xlim(np.min(point_cloud[:,:,1]), np.max(point_cloud[:,:,1]))  # y is <- ->
      ax.set_ylim(np.min(point_cloud[:,:,2]), np.max(point_cloud[:,:,2]))  # z is up
    elif view == 'x-z':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 0], point_cloud[:, :, 2], s=0.75, cmap='gray')
      ax.set_xlim(np.min(point_cloud[:,:,0]), np.max(point_cloud[:,:,0]))  # x is depth
      ax.set_ylim(np.min(point_cloud[:,:,2]), np.max(point_cloud[:,:,2]))  # z is up
    if objects is not None:
      x_values = []
      y_values = []
      z_values = []
      for obj in objects:
        x_values.append(obj[0])
        y_values.append(obj[1])
        if view == '3d':
          z_values.append(obj[2])
      x_values = np.array(x_values)
      y_values = np.array(y_values)
      if view == '3d':
        z_values = np.array(z_values)
        ax.scatter(x_values, y_values, z_values, c='r')
      else:
        ax.scatter(x_values, y_values)

  @staticmethod
  def parse_status(status_file):
    """ Parse a status.json file for a training sequence"""
    with open(status_file, 'r') as s:
      status = json.load(s) 
    max_depth = status["max_depth"]
    # UE world coordinate frame is left-handed XZY (Z up, x forward, y right)
    camera = np.array(list(map(float, status["camera"].split(" "))))
    # x z y offset from world coordinate frame origin
    camera_offset = camera[:3]
    # build camera rotation matrix
    # pitch yaw roll in degrees
    # pitch is rot about Y axis
    # yaw is rot about Z axis
    # roll is rot about X axis
    camera_rot = camera[3:]
    # Pitch 
    rads = (camera_rot[0]) * np.pi/180.
    R1 = np.array([
        [np.cos(rads), 0., np.sin(rads), 0],
        [0, 1, 0, 0],
        [-np.sin(rads), 0, np.cos(rads), 0],
        [0, 0, 0, 1.]
    ])
    # Yaw 
    rads = (camera_rot[1] * np.pi/180)
    R2 = np.array([
        [np.cos(rads), np.sin(rads), 0, 0],
        [-np.sin(rads), np.cos(rads), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # Roll
    rads = (camera_rot[2] * np.pi/180)
    R3 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rads), -np.sin(rads), 0],
        [0, np.sin(rads), np.cos(rads), 0,],
        [0, 0, 0, 1]
    ])
    R = np.matmul(R2, R3)
    R = np.matmul(R1, R)
    R = R[:3, :3]

    # objects is a list of length 100
    # of lists that contain np.arrays for each ball 
    # (xyz) centroid in camera coordinate frame
    objects = []
    for frame in status["frames"]:
      objects.append([])
      for obj in frame:
        if "object" in obj:
          o = frame[obj]
          pos = np.array(list(map(float, o.split(" ")[:3])), ndmin=2)
          pos -= camera_offset  # translation
          pos = np.matmul(R, pos.T).T  # rotation
          objects[-1].append(pos[0])
    return max_depth, objects 

  def backproject_and_rescale(self, obj, depth_ratio):
      """ depth_ratio = new_max_depth / old_max_depth """
      u = obj[1] * self.fx / obj[0]
      v = obj[2] * self.fy / obj[0]
      z = obj[0] * depth_ratio
      return np.array([z, z * u / self.fx, z * v / self.fy])

  def point_2_grid_cell(self, point, scale=1):
    """
    Given an arbitrary point from the point cloud, return
    the i, j coords of its grid cell in the BEV map.
    """
    # in BEV, the x coord maps to the y-dim of point cloud
    i = int(np.floor(point[1] / (self.bev_x_res * scale)))
    j = int(np.floor(point[0] / (self.bev_y_res * scale)))
    return i, j
    
  def grid_cell_2_point(self, i, j, scale=1, k=-1):
    """
    Map grid cell i,j to the point in point cloud
    """
    y = i * self.bev_x_res * scale
    x = j * self.bev_y_res * scale
    if k >= 0:
        z = k * (self.bev_z_dim / self.bev_z_channels) * scale
        return x, y, z
    return x, y
  
  def z_2_grid(self, z, scale=1):
    """ Map z of point to one of the discretized channels """ 
    z_res = (scale * self.bev_z_dim) / self.bev_z_channels
    return int(np.floor(z / z_res))

  def point_cloud_2_BEV(self, point_cloud):
    """
    Given a point cloud of shape (288, 288, 3), map to a BEV binary grid map.
    The map grid cells = 1 if any point maps to the cell, else 0.
    
    Each dimension given by self.bev_dim / self.bev_res
    """
    grid_x_dim = np.ceil(self.bev_x_dim / self.bev_x_res)
    grid_y_dim = np.ceil(self.bev_y_dim / self.bev_y_res)
    bev_z_res = np.ceil(self.bev_z_dim / self.bev_z_channels)
    #grid = np.zeros((int(grid_x_dim), int(grid_y_dim), int(self.bev_z_channels)), dtype=np.uint8)
    # put in CHW format for PyTorch
    grid = np.zeros((int(self.bev_z_channels), int(grid_y_dim), int(grid_x_dim)), dtype=np.uint8)
    # partition by z dimension, and clean up point cloud
    offsets = []
    for i in range(3):
      if np.min(point_cloud[:,:,i]) < 0:
        offsets.append(abs(np.min(point_cloud[:,:,i])))
        point_cloud[:,:,i] += offsets[-1]
      else:
        offsets.append(0.)
    z_start = 0
    for h in range(self.bev_z_channels):
      z_end = z_start + bev_z_res
      pts = point_cloud[(point_cloud[:,:,2] >= z_start) & (point_cloud[:,:,2] < z_end)]
      # compute grid cell for each point
      for p in pts:
        # throw out points at the camera
        if p[0] == 0.0:
          continue
        i, j = self.point_2_grid_cell(p)
        # throw out points outside of the grid
        if i < grid_x_dim and j < grid_y_dim:
          grid[h, j, i] |= 1
      z_start += bev_z_res
    return grid, offsets

if __name__ == '__main__':
  import os
  from scipy.ndimage import imread
  from tqdm import tqdm
  
  #data = '/home/pemami/Workspace/3D-MOT-with-object-permanence/sample_data/train_seq_1_1'
  #max_depth, objects = Depth2BEV.parse_status(os.path.join(data, 'status.json'))
  #depth_data = np.float32(imread(os.path.join(data, 'depth', 'depth_030.png')))
  depth2bev = Depth2BEV()
  #pc = depth2bev.depth_2_point_cloud(depth_data, max_depth)
  #bev, _ = depth2bev.point_cloud_2_BEV(pc)
  #print(bev.shape)
  #print(objects[29])
  train_data_dir = '/media/pemami/DATA/intphys/train'
  train_dirs = os.listdir(train_data_dir)
  #train_dirs = ['00009_block_O1_train']
  # for each sequence, grab the status.json and compute max depth and objects
  for f in tqdm(train_dirs):
    if not os.path.isdir(os.path.join(train_data_dir, f)):
      continue    
    status = os.path.join(train_data_dir, f, 'status.json')
    max_depth, objects = Depth2BEV.parse_status(status)
    # write to txt file
    annot_dir = os.path.join(train_data_dir, f, 'annotations')
    if not os.path.exists(annot_dir):
      os.mkdir(annot_dir)
    # get mask info from status
    with open(status, 'r') as s:
      ss = json.load(s) 
      mask_list = ss['masks_grayscale']
    object_masks = []
    occluder_masks = []
    for m in mask_list:
      if "object" in m[1]:
        object_masks.append(m[0])
      elif "occluder" in m[1]:
        occluder_masks.append(m[0])
    for idx, frame in enumerate(objects):
      annot_file = os.path.join(annot_dir, '%03d.txt' %(idx+1)) 
      print(annot_file) 
      if os.path.exists(annot_file):
        os.remove(annot_file)
      with open(annot_file, 'w') as annot:
        annot.write("{}\n".format(max_depth))
        # grab mask 
        mask = imread(os.path.join(train_data_dir, f, 'mask', 'mask_%03d.png' %(idx+1)))
        for o in frame:
          # Check occlusion
          # backproject to image coordinates
          # Check mask label at image coordinates
          # if mask label is wall and not an object, then skip bc occlusion
          
          # backprojection
          u = int(np.rint(o[1] * depth2bev.fx / o[0]) + depth2bev.cx)
          v = int(depth2bev.cy - np.rint(o[2] * depth2bev.fy / o[0]))
          # check mask label
          if 0 <= u < depth2bev.frame_width and 0 <= v < depth2bev.frame_height:
            mask_label = mask[v, u]
            if mask_label not in occluder_masks:
                annot.write("{} {} {}\n".format(o[0], o[1], o[2]))

    

