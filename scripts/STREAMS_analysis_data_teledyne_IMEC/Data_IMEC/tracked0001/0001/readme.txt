primer on how to work with the processed data. 

overview:

Frames_2: anonymized images
Frames_2_predictions: Predictions are made on RGB data by YOLO, has its own readme.txt (made on non-anonymized images)

PointCloud_1: pointclouds made by Livox
fg_PointCloud_1: extracted foreground (background is given in BGR2.pcd)
fg_predictions: 3d predictions made on extracted foreground by PointRCNN
full_pcd_predictions: 3d predictions made on whole pointcloud by PointRCNN

Streams_labels: contains the GT labels
track_dataframes: dataframes of inferred tracks

calib.json: calibration of ground-plane-aligned(!) lidar to camera (all pointcloud data is aligned with the ground plane already, unless otherwise stated)
Note that this 'diverts' the lidar data from its natural state (ie rays)

a more detailed description is given now.


LIDAR COORDS     ----------------------------------------------------------------------------------------------------------------------------------------------
The LiDAR point cloud data is transformed already, so it is aligned with the ground plane. X points out, Y points to the left, and Z complements, i.e. it points to the sky.
The origin is still 2.16 meters above the ground.

BACKGROUND           ----------------------------------------------------------------------------------------------------------------------------------------------
BGR2.pcd: a pointcloud of the background.


EXTRACTED FOREGROUND ----------------------------------------------------------------------------------------------------------------------------------------------
The extracted foreground pcds are delivered in numpy array format and must be loaded with np.load() in Python.
The origin here is 1.5 meters above ground level. This is to improve the 3d detection. 

Foreground extraction based on background built by MLE.

predictions: X, Y, Z, length, Width, Height, Rotation, Confidence.

Both foreground and full pcd predictions are translated to have the origin 2.16 meters above the ground plane.


LABELS ------------------------------------------------------------------------------------------------------------------------------------------------------------
The label columns are: X_center, Y_center, Z_center, length, width, height, rotation (rad), class, occlusion flag, track ID

That is, we indicate the center of the 3D bounding box. The rotation angle starts from the X axis, going to the Y axis.
Be mindful that an occlusion flag is present. Also, this flag is generally only set to True when the occlusion is 100%. Be aware of these facts when running diagnostics, it will impact your results.


CALIBRATION ------------------------------------------------------------------------------------------------------------------------------------------------------------
In calib, you can find the intrinsic and extrinsic matrix for the ground plane aligned LiDAR to RGB projection.


classes ------------------------------------------------------------------------------------------------------------------------------------------------------------
class predictions in yolo:


    # "0": "person",
    # "1": "bicycle",
    # "2": "car",
    # "3": "motorcycle",

    # "5": "bus",
    # "6": "train",
    # "7": "truck",

    # "9": "traffic light",

    # "11": "stop sign",
    
   
class prediction in pointrcnn:

 {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
 
 
 TRACKING ------------------------------------------------------------------------------------------------------------------------------------------------------------
 In track dataframes, you can find inferred(!) tracks, done by a naive particle filter.
 the columns are named (there is also an index, in front, but this one is meaningless):
 ID,timestamp,x,y,vx,vy,rgb_conf,lidar_conf,frame_index,YOLO_cls,L,T,R,B,X_LiDAR,Y_LiDAR,Z_LiDAR,l,w,h,rot
 
 - ID is a unique track id, the tracker thinks this is an actual identified unique object
 - timestamp corresponds to the numbers used to name the point clouds/images
 - x,y, vx, vy are state estimates done by the filter
 - rgb and lidar conf are confidences from associated detections, if any
 - (!) frame index is not the timestamp, but rather the n-th frame (this should be fixed, the tracker does not consider that the frames are asynchronuous)
 - L T R B originate from the rgb detection if available; if not, they originate from the projected lidar 3D detection if that one is available; otherwise NaN
 - finally the lidar 3d detection info is given, if it is associated to the track at that time; otherwise NaN
 
