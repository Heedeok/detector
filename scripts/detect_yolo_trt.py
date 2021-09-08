#! /usr/bin/env python3

import sys
# import argparse
import os
# import platform
# import shutil
# import time
from pathlib import Path
import cv2
import numpy as np
import message_filters
import random

# ROS
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import Header

## model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yolov5"))
from yolov5.utils.plots import plot_one_box

# ROS messages
from detector.msg import BoundingBox2D, BoundingBox2DArray

# YOLO tensorrt
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yolov5_trt"))
from yolov5_trt.yolo_trt import YoLov5TRT

rospack = rospkg.RosPack()
pkg_pth = rospack.get_path("detector")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class DetectorModel(object):

    def __init__(self):

        self.use_depth = rospy.get_param('~use_depth', 'false')
        # im
        self.img_size = 640

        # Node configuration
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        conf_thres = rospy.get_param('~confidence', 0.4)
        iou_thres = rospy.get_param('~nms_th', 0.5)
        detected_objects_topic = rospy.get_param('~detected_objects')
        published_image_topic = rospy.get_param('~detected_image')
        self.publish_image = rospy.get_param('~publish_image')
    
        # Detection weight initialization
        plugin_library = rospy.get_param('~plugin_library', 'yolov5_trt/engine/libmyplugins.so')
        engine = rospy.get_param('~engine', 'yolov5_trt/engine/yolov5.engine')
        self.plugin_library = os.path.join(pkg_pth, 'scripts', plugin_library)
        self.engine = os.path.join(pkg_pth, 'scripts', engine)
        ctypes.CDLL(self.plugin_library)
        rospy.loginfo("Found detector engine, loading %s", self.engine)

        # COCO index names
        self.names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
        
        # if self.half:
        #     self.detection_model.half()  
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Load Class and initialize
        classes = rospy.get_param('~class_name', None)
        if classes == "None":
            classes = None
        elif classes.find(', ') == -1:
            classes = self.names.index(classes)
        else:
            classes = classes.split(', ')
            classes = [self.names.index(i) for i in classes]
        
        # Detecion model initialzation
        self.detection_model = YoLov5TRT(self.engine, conf_thres, iou_thres, classes)
        # create a new thread to do warm_up
        for i in range(10):
            self.detection_model.infer(self.detection_model.get_raw_image_zeros())
    
        rospy.loginfo("Yolov5 TRT warm up end!")

        # ROS publish topic
        self.topic_header = None
        self.topic_header_stamp = None

        # ROS Subscriber synchronized color and depth
        if self.use_depth:
            self.image_sub = message_filters.Subscriber(image_topic,Image)
            self.depth_sub = message_filters.Subscriber(depth_topic,Image)
            ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 1)
            ts.registerCallback(self.callback)
        else:
            self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)


        # ROS Publisher
        self.pub_detector_ = rospy.Publisher(detected_objects_topic, BoundingBox2DArray, queue_size=10)
        if self.publish_image : 
            self.pub_viz_ = rospy.Publisher(published_image_topic, Image, queue_size=10)
        
        rospy.loginfo("Launched node for object detector")
            
        rospy.spin()

    def detect(self, input_image, color_img, depth_img=None):

        # Run inference
        t0 = rospy.get_time()

        img, im0 = input_image, color_img

        # Detection
        det, time = self.detection_model.infer(img, im0.shape[0],im0.shape[1])
        det = np.array(det)
        # print("np pred : ", det.shape)
        rospy.loginfo("Detecion inference time : (%.3fs),(%.3f Hz)"%(time, 1/(time)))
        
        # Detection process
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string

        if det is not None and len(det):
            # Print results
            tmp = np.array(det[:, -1]).flatten()
            for c in tmp:
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            bbox_xywh = []
            confs = []
            labels = []

            for box in det:
                x_c, y_c, bbox_w, bbox_h = self.bbox_rel(box[:4])
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append(box[4])

                if self.publish_image:
                    labels.append(self.names[int(box[-1])])
                    label = f'{self.names[int(box[-1])]} {box[4]:.2f}'
                    plot_one_box(box[:4], im0, label=label, color=self.colors[int(box[-1])], line_thickness=3)
                    # self.plot_one_box(box[:4], im0, label=label)

            if self.use_depth:
                self.inference_publish(im0, bbox_xywh, confs, labels, depth_img)
            else:
                self.inference_publish(im0, bbox_xywh, confs, labels)

        
        t3 = rospy.get_time()
        rospy.loginfo('%sDone. (%.3fs),(%.3f Hz)' % (s, t3 - t0, 1/(t3 - t0)))
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        """
        description: Plots one bounding box on image img,
                    this function comes from YoLov5 project.
        param: 
            x:      a box likes [x1,y1,x2,y2]
            img:    a opencv image object
            color:  color to draw rectangle, such as (0,255,0)
            label:  str
            line_thickness: int
        return:
            no return

        # """
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


    def inference_publish(self, vis_image, bbox, confidence, identities=None, depth_img=None):
        # Publish Bbox topic
        output_boxes = BoundingBox2DArray()
        output_boxes.header = self.topic_header
        output_boxes.header.stamp = self.topic_header_stamp
	
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            
            
            center_x = x1
            center_y = y1
            height = y2
            width = x2
            score = confidence[i]

            distance = []
            distance.append(0)

            tmp_box = BoundingBox2D()
            tmp_box.center_x = int(center_x) # about width
            tmp_box.center_y = int(center_y) # about height
            tmp_box.height = int(height)
            tmp_box.width = int(width)
            tmp_box.score = float(score)
            tmp_box.id = str(identities[i]) if identities is not None else 0

            start = rospy.get_time()
            if depth_img is not None:
                # calculate the distance using median filter
                #if height >= width:
                #    kernel = width
                #else: kernel = height
                kernel = 10
                for dx in range(-kernel, kernel+1):
                    for dy in range(-kernel, kernel+1):
                        x = center_x + dx 
                        y = center_y + dy 
                        if (x < 0 )or (y < 0) or (y >= depth_img.shape[0]) or (x >= depth_img.shape[1]):
                            continue
                        
                        distance.append(int(depth_img[y][x]))
            distance.sort()
            tmp_box.distance = distance[len(distance)//2]	
            end = rospy.get_time()


            #tmp_box.distance = int(statistics.median(distance))
            output_boxes.boundingboxes.append(tmp_box)
        
        if self.publish_image:

            # Publisch vis_image topic
            image_temp=Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = 'map'
            image_temp.height=vis_image.shape[0]
            image_temp.width=vis_image.shape[1]
            image_temp.encoding='rgb8'
            image_temp.data=np.array(vis_image).tostring()
            image_temp.header=header
            image_temp.step=1241*3

            self.pub_viz_.publish(image_temp)
        
        # Publish the tracker topic
        self.pub_detector_.publish(output_boxes)
 
    def image_raw_to_input_image(self, input_image):

        # input_image = letterbox(raw_image, new_shape=self.img_size)[0]
        # print("input img : ", input_image.shape)
        h, w, c = input_image.shape

        r_w = self.img_size / w
        r_h = self.img_size / h

        if r_h > r_w:
            tw = self.img_size
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.img_size - th) / 2)
            ty2 = self.img_size - th - ty1
        else:
            tw = int(r_h * w)
            th = self.img_size
            tx1 = int((self.img_size - tw) / 2)
            tx2 = self.img_size - tw - tx1
            ty1 = ty2 = 0
    
        input_image = np.pad(input_image, ((ty1, ty2),(tx1,tx2),(0,0)), 'constant', constant_values=0)
    
        # Convert
        input_image = input_image.astype(np.float32)
        input_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        input_image = np.transpose(input_image, [2, 0, 1])
        # CHW to NCHW format
        input_image = np.expand_dims(input_image, axis=0)
        input_image = np.ascontiguousarray(input_image)

        return input_image

    def bbox_rel(self, xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        """ xyxy = (x1, y1, x2, y2)  -> (x_C, y_c, w, h)"""
        # print('xyxy : [{}, {}, {}, {}]'.format(xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h
        # return y_c, x_c, w, h

    def callback(self, color, depth):
        rospy.loginfo('image received!')

        try:
            color_image = np.frombuffer(color.data, dtype=np.uint8).reshape(color.height, color.width, -1)
            depth_image = np.frombuffer(depth.data, dtype=np.int16).reshape(depth.height, depth.width, -1)
        except  Exception as e:
            print(e)
            rospy.loginfo("image callback error")   
            return False
        
        self.topic_header = color.header
        self.topic_header_stamp = color.header.stamp 

        input_image = self.image_raw_to_input_image(color_image)
        
        self.detect(input_image, color_image, depth_image)
        
        rospy.loginfo('inference end!')
    
    def image_callback(self, image):
        rospy.loginfo('image received!')

        try:
            color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        except Exception as e:
            print(e)
            rospy.loginfo("image callback error")   
            return False

        self.topic_header = image.header
        self.topic_header_stamp = image.header.stamp 

        input_image = self.image_raw_to_input_image(color_image)
        
        self.detect(input_image, color_image)
        
        rospy.loginfo('inference end!')

def main():

    # Initialize node
    rospy.init_node('detector_node')
    
    # Define traker
    detector_node = DetectorModel()


if __name__ == '__main__':
    main()
