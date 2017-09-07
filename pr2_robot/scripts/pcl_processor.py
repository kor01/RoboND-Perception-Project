import pcl
import rospy
import pickle
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from sklearn.preprocessing import LabelEncoder
import pcl_helper as ph
from sensor_stick.srv import GetNormals

LEAVE_SIZE = 0.005


def downsample_pcl(data):
  vox = data.make_voxel_grid_filter()
  vox.set_leaf_size(LEAVE_SIZE, LEAVE_SIZE, LEAVE_SIZE)
  return vox.filter()


def workspace_filter(data):
  passthrough = data.make_passthrough_filter()
  passthrough.set_filter_field_name('z')
  passthrough.set_filter_limits(0.61, 1.2)
  data = passthrough.filter()
  passthrough = data.make_passthrough_filter()
  passthrough.set_filter_field_name('x')
  passthrough.set_filter_limits(0.31, 2)
  return passthrough.filter()

MAX_DISTANCE = 0.005


def segment_table_objects(data):
  seg = data.make_segmenter()
  seg.set_method_type(pcl.SAC_RANSAC)
  seg.set_model_type(pcl.SACMODEL_PLANE)
  seg.set_distance_threshold(MAX_DISTANCE)
  inliers, _ = seg.segment()
  table = data.extract(inliers)
  objects = data.extract(inliers, negative=True)
  return table, objects


def cluster_objects(objects):
  objects = ph.XYZRGB_to_XYZ(objects)
  tree = objects.make_kdtree()
  ec = objects.make_EuclideanClusterExtraction()
  ec.set_ClusterTolerance(0.01)
  ec.set_MinClusterSize(125)
  ec.set_MaxClusterSize(10000)
  ec.set_SearchMethod(tree)
  ret = ec.Extract()
  return ret


def visualize_objects(cloud, clusters, color=True):
  cluster_color = ph.get_color_list(len(clusters))
  ret = []
  for j, indices in enumerate(clusters):
    for i, indice in enumerate(indices):
      if color:
        c = ph.rgb_to_float(cluster_color[j])
      else:
        c = cloud[indice][3]
      ret.append([cloud[indice][0], cloud[indice][1],
                  cloud[indice][2], c])
  pc = pcl.PointCloud_PointXYZRGB()
  pc.from_list(ret)
  return pc


def index_point_cloud(cluster_idx, cloud):
  ret = []
  for i in cluster_idx:
    ret.append(cloud[i])
  pc = pcl.PointCloud_PointXYZRGB()
  pc.from_list(ret)
  return pc


def get_normals(cloud):
  get_normals_prox = rospy.ServiceProxy(
    '/feature_extractor/get_normals', GetNormals)
  return get_normals_prox(cloud).cluster


def remove_outliers(cloud):
  outlier_filter = cloud.make_statistical_outlier_filter()
  outlier_filter.set_mean_k(100)
  outlier_filter.set_std_dev_mul_thresh(0.1)
  cloud = outlier_filter.filter()
  return cloud


def compute_normal_histograms(norm, bins):
  hist_1 = np.histogram(norm[:, 0], bins=bins, range=(-1.0, 1.0))
  hist_2 = np.histogram(norm[:, 1], bins=bins, range=(-1.0, 1.0))
  hist_3 = np.histogram(norm[:, 2], bins=bins, range=(-1.0, 1.0))
  normed_features = np.concatenate(
    (hist_1[0], hist_2[0], hist_3[0])).astype('float64')
  return normed_features


def compute_color_histograms(cloud, bins):
  colors = cloud[:, 3:].reshape(1, -1, 3).astype('uint8')
  colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV_FULL)
  colors = colors.reshape((-1, 3)).astype('float64')
  channel_1_vals = colors[:, 0]
  channel_2_vals = colors[:, 1]
  channel_3_vals = colors[:, 2]
  hist_1 = np.histogram(channel_1_vals, bins=bins, range=(0, 256))
  hist_2 = np.histogram(channel_2_vals, bins=bins, range=(0, 256))
  hist_3 = np.histogram(channel_3_vals, bins=bins, range=(0, 256))
  hist_features = np.concatenate(
    (hist_1[0], hist_2[0], hist_3[0])).astype('float64')
  hist_features = hist_features / hist_features.sum()
  return hist_features


def normal_to_np(normal_cloud):
  ret = []
  for norm_component in pc2.read_points(
      normal_cloud,
      field_names=('normal_x', 'normal_y', 'normal_z'),
      skip_nans=True):
      ret.append(norm_component[:3])
  return np.array(ret, dtype=np.float64)


def extract_features(
    cloud, norm, norm_bins, color_bins):
  color_hist = compute_color_histograms(cloud, color_bins)
  norm_hist = compute_normal_histograms(norm, norm_bins)
  feature = np.concatenate((color_hist, norm_hist))
  return feature


def cloud_to_np(cloud):
  arr = cloud.to_array()
  obj = [np.concatenate(
    (x[:3], ph.float_to_rgb(x[3]))) for x in arr]
  obj = np.array(obj).astype(np.float64)
  return obj


class RecognitionModel(object):

  def __init__(self, path):
    self._model = pickle.load(open(path, 'rb'))
    self._clf = self._model['classifier']
    self._encoder = LabelEncoder()
    self._encoder.classes_ = self._model['classes']
    self._scaler = self._model['scaler']
    self._norm_bins = self._model['norm_bins']
    self._color_bins = self._model['color_bins']

  def classify(self, cloud):
    norm = get_normals(ph.pcl_to_ros(cloud))
    norm = normal_to_np(norm)
    cloud = cloud_to_np(cloud)
    feature = extract_features(
      cloud, norm, self._norm_bins, self._color_bins)
    prediction = self._clf.predict(
      self._scaler.transform(feature.reshape(1, -1)))
    label = self._encoder.inverse_transform(prediction)[0]
    return label, feature
