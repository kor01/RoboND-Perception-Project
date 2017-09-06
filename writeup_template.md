## Project: Perception Pick & Place

---

#### Run The Project:

1. clone the project to `catkin_ws/src/`

2. `catkin_make` in workspace root
3. `roslaunch pr2_robot pick_place_project.launch`
4. `rosrun pr2_robot perception_main.py`
5. output yaml in directory `pr2_robot/yaml_outputs`


#### Required Steps for a Passing Submission:

1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).

2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

#### Extra Challenges: Complete the Pick & Place
7. not yet implemented

---

### [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

---
#### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

#### Exercise 1, 2 and 3 pipeline implemented
1. voxel-downsampling, pass-through workspace extraction, RANSAC worspace extraction is implemented in file:

    a. `pr2_robot/script/pcl_processor.py: 14 downsample_pcl()`

    b. `pr2_robot/script/pcl_processor.py: 20 workspace_filter()`

    c. `pr2_robot/script/pcl_processor.py: segment_table_objects()`


2. Euclidean clustering segmentation is implemented in file (`pr2_robot/script/pcl_processor.py: 43 cluster_objects`)

3. SVM training is implemented in: `pr2_robot/notebook/cloud_recognition.ipynb`


#### Pick and Place Setup

1. the ros-node in implemented in `pr2_robot/script/perception_main.py: 163 main()`

2. feature extraction function is implemented in `pr2_robot/script/pcl_processor.py: 129 extract_features()`

3. svm inference is implemented in `pr2_robot/script/pcl_processor.py: 145 class RecognitionModel`


#### Perception Algorithm:

1. use 3 different models for 3 worlds to maximize the recognition precision

2. use different histogram bin-size for 3 worlds to maximize the performance of svm model

3. use `cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)` to replace matplotlib hsv implementation to get much faster feature extraction speed

4. I modified `capture_feature.py` script to dump the raw cloud and norm data to be able to experiment feature extraction algorithm faster (without going through feature capture every time), the new script is included in `pr2_robot/script/capture_feature.py`

5. the future improvement will be the implementation of the extra challenges
