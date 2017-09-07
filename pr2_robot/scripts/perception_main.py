#!/usr/bin/env python

import os
import sys

# to be able to import pcl_helper and pc_processor
dirname = os.path.dirname(__file__)
sys.path.append(dirname)

import rospy
import yaml
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_stick.msg import DetectedObject
from sensor_stick.msg import DetectedObjectsArray
from pr2_robot.srv import PickPlaceRequest
import pcl_helper as ph
import pcl_processor as pp
from marker_tools import make_label
from pr2_robot.srv import PickPlace
from rospy_message_converter.message_converter \
  import convert_ros_message_to_dictionary as cvt


def make_yaml_dict(req):
  assert isinstance(req, PickPlaceRequest)
  yaml_dict = {}
  yaml_dict["test_scene_num"] = req.test_scene_num.data
  yaml_dict["arm_name"] = req.arm_name.data
  yaml_dict["object_name"] = req.object_name.data
  yaml_dict["pick_pose"] = cvt(req.pick_pose)
  yaml_dict["place_pose"] = cvt(req.place_pose)
  return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
  data_dict = {"object_list": dict_list}
  with open(yaml_filename, 'w') as outfile:
    yaml.dump(data_dict, outfile, default_flow_style=False)


def pcl_callback(obj_pub, table_pub,
                 marker_pub, detect_pub,
                 debug_pub, msg, model, mover, add_idx=False):

  data = ph.ros_to_pcl(msg)
  data = pp.downsample_pcl(data)
  data = pp.remove_outliers(data)

  workspace = pp.workspace_filter(data)
  table, objects = pp.segment_table_objects(workspace)
  table_pub.publish(ph.pcl_to_ros(table))
  clusters = pp.cluster_objects(objects)
  visual_objects = pp.visualize_objects(objects, clusters, color=True)
  obj_pub.publish(ph.pcl_to_ros(visual_objects))

  labels, centroids, dos, object_list, features = [], [], [], [], []
  for index, cluster in enumerate(clusters):
    cloud = pp.index_point_cloud(cluster, objects)
    object_list.append(cloud)
    centroids.append(cloud.to_array().mean(axis=0)[:3])
    label, feature = model.classify(cloud)
    labels.append(label)
    features.append(feature)
    obj = DetectedObject()
    obj.label = label
    obj.cloud = ph.pcl_to_ros(cloud)
    dos.append(obj)
    label_pos = list(cloud[0])[:3]
    label_pos[2] += .4

    if add_idx:
      label + ':' % index
    marker_pub.publish(
      make_label(label, label_pos, index))

  '''
  for i, d in enumerate(objects):
    print 'obj-%d' % i, d.to_array()[:, :3].min(axis=0)

  print 'max_x', objects[-1].to_array()[:, :3].max(axis=0)
  '''

  detect_pub.publish(dos)
  mover(centroids, labels)


def pr2_mover(centers, labels, dropbox, target_list, scene_id):

  group_map = {x['group']: x for x in dropbox}
  labels = {x: i for i, x in enumerate(labels)}
  dict_list = []

  for target in target_list:
    name = target['name']
    if name not in labels:
      rospy.logwarn('unable to find %s', name)
      continue
    center = centers[labels[name]]
    box = group_map[target['group']]
    arm_name, place_pos = box['name'], box['position']
    req = PickPlaceRequest()
    req.arm_name.data = arm_name
    req.test_scene_num.data = scene_id
    req.object_name.data = name
    pos = req.pick_pose.position
    pos.x, pos.y, pos.z = [np.asscalar(x) for x in center]
    pos = req.place_pose.position
    pos.x, pos.y, pos.z = place_pos
    dict_list.append(make_yaml_dict(req))
    
    
    try:
      rospy.wait_for_service('pick_place_routine')
      pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
      resp = pick_place_routine(req)
      print "Response: ", resp.success
    except rospy.ServiceException, e:
      print "Service call failed: %s" % e
    

  send_to_yaml('output_%d.yaml' % scene_id, dict_list)
  print 'output_%d.yaml ok' % scene_id


def main():

  targets = rospy.get_param('/object_list')
  dropbox = rospy.get_param('/dropbox')

  if len(targets) == 3:
    scene_id = 1
  elif len(targets) == 5:
    scene_id = 2
  else:
    scene_id = 3

  def mover(x, y):
    return pr2_mover(
      x, y, dropbox, targets, scene_id)

  classifier_dir = os.path.join(dirname, 'classifiers')
  model_path = 'svm_model_3.sav'
  model_path = os.path.join(classifier_dir, model_path)
  
  model = pp.RecognitionModel(model_path)

  rospy.init_node('clustering', anonymous=True)
  objects_pub = rospy.Publisher('/pcl_objects', pc2.PointCloud2, queue_size=1)
  table_pub = rospy.Publisher('/pcl_table', pc2.PointCloud2, queue_size=1)
  detect_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)
  marker_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
  debug_pub = rospy.Publisher('/object_debug', pc2.PointCloud2, queue_size=1)
  _ = rospy.Subscriber(
    '/pr2/world/points',
    pc2.PointCloud2, lambda x: pcl_callback(
      objects_pub, table_pub, marker_pub,
      detect_pub, debug_pub, x, model, mover), queue_size=1)

  rospy.spin()

if __name__ == '__main__':
    main()
