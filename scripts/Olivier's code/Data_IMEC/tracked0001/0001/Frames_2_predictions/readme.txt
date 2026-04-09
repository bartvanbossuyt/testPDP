TripleTile meaning: the 1280x720 image (the 24px header not included) is cut in three 640x640 images. and cutting 40 above and below (again, header not included). the left image is 0 to 640, middle is 320 to 960, right image 640 to 1280. thus, the overlaps are 320 wide. For the left image, all centers left of 480 are kept. the ones to the right are removed, as they should be better detected in the middle image. An analog approach is taken for the middle (480 to 800) and right (800, 1280) images.

Results are generated for different IoUs at these partial images (not among).
eg 0.2: bounding boxes overlapping more than 20% are combined.

the higher the minimum IoU threshold, the more predictions.

the confidence threshold is 0.0001, included classes are :

    # "0": "person",
    # "1": "bicycle",
    # "2": "car",
    # "3": "motorcycle",

    # "5": "bus",
    # "6": "train",
    # "7": "truck",

    # "9": "traffic light",

    # "11": "stop sign",


--------------------------------------------------------------------------------------------------------


code as of jan 27 2026 was (bb_intersection_over_union function is not used here):

from ultralytics import YOLO

import os
import cv2
import matplotlib.pyplot as plt

import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # L T R B
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0, 0, 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    unionArea =  float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea

    # return the intersection over union value
    return iou, interArea/boxAArea, unionArea
# classes
# classes
    # "0": "person",
    # "1": "bicycle",
    # "2": "car",
    # "3": "motorcycle",
    # "4": "airplane",
    # "5": "bus",
    # "6": "train",
    # "7": "truck",
    # "8": "boat",
    # "9": "traffic light",
    # "10": "fire hydrant",
    # "11": "stop sign",
    # "12": "parking meter",
    # "13": "bench",
    # "14": "bird",
    # "15": "cat",
    # "16": "dog",
    # "17": "horse"

image_folder = "/run/user/2308/gvfs/sftp:host=ipids/scratch/jarmalfl/Streams_unblurred/0001/Frames_2"

image_paths = [os.path.join(image_folder, i) for i in os.listdir(image_folder)]
image_paths.sort()

"""!!! folder name !!!"""
save_folder = "/run/user/2308/gvfs/sftp:host=ipids/scratch/jarmalfl/Streams_unblurred/0001/Frames_2_preds/"

model_names = ['yolo11n', 'yolo11m', 'yolo11x']
ious = [.2, .6, .8, .9, .95]

for model_name in model_names:

    model = YOLO(model_name + '.pt')

    for iou in ious:

        model_folder = os.path.join(save_folder, model_name, 'IoU_'+str(iou).replace('.','p'))

        os.makedirs(model_folder, exist_ok=True)

        for image_path in image_paths:
            im_name = image_path.split('/')[-1].split('.')[0]
            save_path = os.path.join(model_folder, im_name+'.txt')

            im = cv2.imread(image_path)
            L, M, R = im[40+24:-40,:640], im[40+24:-40,320:-320], im[40+24:-40,-640:]

            bounds = [(0, 480), (160, 480), (160, 640)]
            offsets = [0, 320, 640]

            image_preds = []
            for sub_im, bound, offset in zip([L, M, R], bounds, offsets):

                result = model.predict(sub_im, stream=False, conf=0.0001, save_conf=False, save_txt=False, max_det=10000,
                                                  iou=iou, classes=[0, 1, 2, 3, 5, 6, 7, 9, 11])[0]

                boxes = result.boxes.cpu().numpy()
                xywh = boxes.xywh
                ltrb = boxes.xyxy
                confidences = boxes.conf
                classes = boxes.cls

                centers = xywh[:,0]
                valid = (bound[0]<centers)&(centers<bound[1])
                valid_ltrb, valid_confidences, valid_classes = ltrb[valid], confidences[valid], classes[valid]

                valid_ltrb[:, 0] += offset
                valid_ltrb[:, 2] += offset

                valid_ltrb[:, 1] += 40+24
                valid_ltrb[:, 3] += 40+24

                save2txt = np.hstack([valid_ltrb, valid_confidences[:, np.newaxis], valid_classes[:, np.newaxis]])
                image_preds.append(save2txt)

            save2txts = np.concat(image_preds)

            ind = np.lexsort((1-save2txts[:,-2], save2txts[:,-1]))
            save2txts = save2txts[ind]

            np.savetxt(save_path, save2txts, fmt=['%.i','%.i', '%.i', '%.i', '%1.6f', '%.i'])


