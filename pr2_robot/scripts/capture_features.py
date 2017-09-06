#!/usr/bin/env python
import numpy as np
import pickle
import rospy
import pcl

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.features import normal_to_np
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def remove_outliers(cloud):
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(100)
    outlier_filter.set_std_dev_mul_thresh(0.1)
    cloud = outlier_filter.filter()
    return cloud

LEAVE_SIZE = 0.005


def downsample_pcl(data):
    vox = data.make_voxel_grid_filter()
    vox.set_leaf_size(LEAVE_SIZE, LEAVE_SIZE, LEAVE_SIZE)
    return vox.filter()


if __name__ == '__main__':
    rospy.init_node('capture_node')

    models = [
       'sticky_notes',
       'book',
       'snacks',
       'biscuits',
       'eraser',
       'soap2',
        'soap', 'glue']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(150):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 10:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()
                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            sample_cloud_pcl = ros_to_pcl(sample_cloud)
            sample_cloud_pcl = downsample_pcl(sample_cloud_pcl)
            #sample_cloud_pcl = remove_outliers(sample_cloud_pcl)
            sample_cloud_arr = sample_cloud_pcl.to_array()
            sample_cloud = pcl_to_ros(sample_cloud_pcl)
            # Extract histogram features
            #chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            normals = normal_to_np(normals)
            #nhists = compute_normal_histograms(normals)
            #feature = np.concatenate((chists, nhists))
            if sample_was_good:
                obj = [np.concatenate((x[:3], float_to_rgb(x[3]))) for x in sample_cloud_arr]
                obj = np.array(obj).astype(np.float64)
                
            labeled_features.append((obj, normals, model_name))

        delete_model()
    
    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

