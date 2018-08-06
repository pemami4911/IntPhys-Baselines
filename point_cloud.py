import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
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
  def __init__(self, opt, image_dims=[288, 288], pc_dims=[2500, 3480, 800], 
      fovx=90, fovy=90, grid_x_res=10, grid_y_res=10, grid_z_res=10, fixed_depth=2500):
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
    self.grid_x_res = grid_x_res
    self.grid_y_res = grid_y_res
    self.grid_z_res = grid_z_res
    self.pc_x_dim = pc_dims[0]
    self.pc_y_dim = pc_dims[1]
    self.pc_z_dim = pc_dims[2]
    self.bev_x_dim = opt.view_dims['BEV'][0]
    self.bev_y_dim = opt.view_dims['BEV'][1]
    self.grid_height_channels = opt.view_dims['BEV'][2]
    self.fv_x_dim = opt.view_dims['FV'][0]
    self.fv_y_dim = opt.view_dims['FV'][1]
    self.grid_depth_channels = opt.view_dims['FV'][2]
    self.fixed_depth = fixed_depth
    self.ball_radius = opt.ball_radius
    self.regression_stats = []
    if opt.normalize_regression:
        with open(opt.regression_statistics_file, 'r') as rs:
            for i in range(3):
                l = rs.readline().split(" ")
                md = int(l[0])
                mean = float(l[1])
                std = np.sqrt(float(l[2]))
                self.regression_stats.append([mean, std])


  def depth_2_point_cloud(self, depth_data, max_depth=None):
    if not max_depth:
      max_depth = self.fixed_depth
    depth_data = (depth_data * max_depth) / np.max(np.abs(depth_data))
    point_cloud = np.zeros((int(self.frame_width/2), int(self.frame_height/2), 3), dtype=np.float32)
    for r in range(0, self.frame_height, 2):
      for c in range(0, self.frame_width, 2):
        if depth_data[r][c] > 0.0:
          v = -(r - self.cy)
          u = (c - self.cx)
          Z = depth_data[r][c]
          # Unreal engine uses left-hand XZY coordinates with z-up
          point_cloud[int(np.ceil(r/2)), int(np.ceil(c/2)), :] = [Z, u*Z/self.fx, v*Z/self.fy]
    return point_cloud

  @staticmethod 
  def display_point_cloud(point_cloud, view='3d', objects=None, radius=200, save=False, name=None):
    def draw_sphere(cx, cy, cz):
      u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
      x = (radius*np.cos(u)*np.sin(v)) + cx
      y = (radius*np.sin(u)*np.sin(v)) + cy
      z = (radius*np.cos(v)) + cz
      ax.plot_wireframe(x, y, z, color="r")

    if view=='3d':
      # flattened pc for plotting
      point_cloud_flt = np.reshape(point_cloud, (point_cloud.shape[0] * point_cloud.shape[1], 3))
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111, projection='3d')  
      ax.scatter(point_cloud[:, :, 0], point_cloud[:, :, 1], point_cloud[:, :, 2], s=0.75, c=point_cloud_flt[:, 2], cmap='gray')
      ax.set_xlim(0, 2500)
      ax.set_ylim(0, 3480)
      ax.set_zlim(0, 800)
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")
    elif view == 'x-y':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 1], point_cloud[:, :, 0], s=0.75, cmap='gray')
      ax.set_xlim(point_cloud[:,:,1].min(), point_cloud[:,:,1].max())  # y is <- ->
      ax.set_ylim(point_cloud[:,:,0].min(), point_cloud[:,:,0].max())  # x is depth
    elif view == 'y-z':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 1], point_cloud[:, :, 2], s=0.75, cmap='gray')
      ax.set_xlim(point_cloud[:,:,1].min(), point_cloud[:,:,1].max())  # y is <- ->
      ax.set_ylim(point_cloud[:,:,2].min(), point_cloud[:,:,2].max())  # z is up
    elif view == 'x-z':
      f = plt.figure(figsize=(8,8))
      ax = f.add_subplot(111) 
      plt.scatter(point_cloud[:, :, 0], point_cloud[:, :, 2], s=0.75, cmap='gray')
      ax.set_xlim(point_cloud[:,:,0].min(), point_cloud[:,:,0].max())  # x is depth
      ax.set_ylim(point_cloud[:,:,2].min(), point_cloud[:,:,2].max())  # z is up
    if objects is not None:
      if view != '3d':
          x_values = []
          y_values = []
          for obj in objects:
            x_values.append(obj[0])
            y_values.append(obj[1])
          x_values = np.array(x_values)
          y_values = np.array(y_values)
          ax.scatter(x_values, y_values)
      else:
        for obj in objects:
            draw_sphere(obj[0], obj[1], obj[2])
    ax.view_init(15, 200)
    if save:
      plt.savefig(name)
    plt.close()

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
    occluders = []
    for frame in status["frames"]:
      objects.append([])
      occluders.append([])
      for obj in frame:
        if "object" in obj:
          o = frame[obj]
          pos = np.array(list(map(float, o.split(" ")[:3])), ndmin=2)
          pos -= camera_offset  # translation
          pos = np.matmul(R, pos.T).T  # rotation
          objects[-1].append(pos[0])
        if "occluder" in obj:
          o = frame[obj]
          pos = np.array(list(map(float, o.split(" ")[:3])), ndmin=2)
          pos -= camera_offset  # translation
          pos = np.matmul(R, pos.T).T  # rotation
          occluders[-1].append(pos[0])
        
    return max_depth, objects, occluders 

  def backproject_and_rescale(self, obj, depth_ratio):
      """ depth_ratio = new_max_depth / old_max_depth """
      u = obj[1] * self.fx / obj[0]
      v = obj[2] * self.fy / obj[0]
      z = obj[0] * depth_ratio
      return np.array([z, z * u / self.fx, z * v / self.fy])

  def point_2_grid_cell(self, point, scale=1, view='BEV'):
    """
    Given an arbitrary point from the point cloud, return
    the i, j coords of its grid cell in the BEV map.
    """
    # in BEV, the x coord maps to the y-dim of point cloud
    if view == 'BEV':
        i_idx = 1; j_idx = 0
    elif view == 'FV':
        i_idx = 1; j_idx = 2
    i = int(np.floor(point[i_idx] / (self.grid_x_res * scale)))
    j = int(np.floor(point[j_idx] / (self.grid_y_res * scale)))
    return i, j
    
  def grid_cell_2_point(self, i, j, scale=1, k=-1, view='BEV'):
    """
    Map grid cell i,j to the point in point cloud
    """
    if view == 'BEV':
        y = i * self.grid_x_res * scale
        x = j * self.grid_y_res * scale
        if k >= 0:
            z = k * (self.pc_z_dim / self.grid_height_channels) * scale
            return x, y, z
        return x, y
    elif view == 'FV':
        y = i * self.grid_x_res * scale
        z = j * self.grid_z_res * scale
        if k >= 0:
            z = k * (self.pc_x_dim / self.grid_depth_channels) * scale
            return x, y, z
        return y, z
  
  def z_2_grid(self, z, scale=1, view='BEV'):
    """ Map z of point to one of the discretized channels """ 
    if view == 'BEV':
        z_res = (scale * self.pc_z_dim) / self.grid_height_channels
    elif view == 'FV':
        z_res = (scale * self.pc_x_dim ) / self.grid_depth_channels
    return int(np.floor(z / z_res))

  def point_cloud_2_view(self, point_cloud, view='BEV'):
    """
    Given a point cloud of shape (144, 144, 3), map to a BEV binary grid map.
    The map grid cells = 1 if any point maps to the cell, else 0.
    
    Each dimension given by self.bev_dim / self.bev_res
    """
    if view == 'BEV':
      # put in CHW format for PyTorch
      grid = np.zeros((self.grid_height_channels, self.bev_y_dim, self.bev_x_dim), dtype=np.uint8)
    elif view == 'FV':
      grid = np.zeros((self.grid_depth_channels, self.fv_y_dim, self.fv_x_dim), dtype=np.uint8)
    # partition by z dimension, and clean up point cloud
    offsets = []
    for i in range(3):
      if np.min(point_cloud[:,:,i]) < 0:
        offsets.append(abs(np.min(point_cloud[:,:,i])))
        point_cloud[:,:,i] += offsets[-1]
      else:
        offsets.append(0.)
    z_start = 0
    if view == 'BEV':
        p_idx = 2
    elif view == 'FV':
        p_idx = 0
    # TODO: Parallelize this for loop
    iters = self.grid_height_channels if view == 'BEV' else self.grid_depth_channels
    for h in range(iters):
      z_end = z_start + self.grid_z_res
      pts = point_cloud[(point_cloud[:,:,p_idx] >= z_start) & (point_cloud[:,:,p_idx] < z_end)]
      # compute grid cell for each point
      # TODO: Parallelize or vectorize?
      for p in pts:
        # throw out points at the camera
        if p[0] == 0.0:
          continue
        i, j = self.point_2_grid_cell(p, view=view)
        # throw out points outside of the grid
        if i < grid.shape[2] and j < grid.shape[1]:
          grid[h, j, i] |= 1
      z_start += self.grid_z_res
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

    

