#! /usr/bin/env python3


import sys
# import argparse
import os
# import platform
# import shutil
# import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import statistics
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
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_one_box

# ROS messages
from detector.msg import BoundingBox2D, BoundingBox2DArray

rospack = rospkg.RosPack()
pkg_pth = rospack.get_path("detector")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class DetectorModel(object):

    def __init__(self):
        
        # set True to speed up constant image size inference
        cudnn.benchmark = True 

        self.use_depth = rospy.get_param('~use_depth', 'false')

        # Node configuration
        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.conf_thres = rospy.get_param('~confidence', 0.4)
        self.iou_thres = rospy.get_param('~nms_th', 0.5)
        self.detected_objects_topic = rospy.get_param('~detected_objects')
        self.published_image_topic = rospy.get_param('~detected_image')
        self.img_size = rospy.get_param('~img_size', 640)
        self.publish_image = rospy.get_param('~publish_image')
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.augment = rospy.get_param('~augment', 'false') # data augmentation
        self.agnostic_nms = rospy.get_param('~agnostic_nms', 'false') # class-agnostic NMS

        # Gpu setting
        self.device = select_device(str(self.gpu_id))
        self.half = self.device.type != 'cpu' 

        # Detection weight initialization
        weight_name = rospy.get_param('~weights_name', 'yolov5/weights/yolov5s.pt')
        self.detecion_weight = os.path.join(pkg_pth, 'scripts', weight_name)
        rospy.loginfo("Found detector weights, loading %s", self.detecion_weight)

        # Detecion model initialzation
        self.detection_model = attempt_load(self.detecion_weight, map_location=self.device)
        self.names = self.detection_model.module.names if hasattr(self.detection_model, 'module') else self.detection_model.names
        if self.half:
            self.detection_model.half()  
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Load Class and initialize
        self.classes = rospy.get_param('~class_name', None)
        if self.classes == "None":
            self.classes = None
        elif self.classes.find(', ') == -1:
            self.classes = self.names.index(self.classes)
        else:
            self.classes = self.classes.split(', ')
            self.classes = [self.names.index(i) for i in self.classes]

        # ROS publish topic
        self.topic_header = None
        self.topic_header_stamp = None

        # ROS Subscriber synchronized color and depth
        if self.use_depth:
            self.image_sub = message_filters.Subscriber(self.image_topic,Image)
            self.depth_sub = message_filters.Subscriber(self.depth_topic,Image)
            ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 1)
            ts.registerCallback(self.callback)
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)


        # ROS Publisher
        self.pub_detector_ = rospy.Publisher(self.detected_objects_topic, BoundingBox2DArray, queue_size=10)
        if self.publish_image : 
            self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        
        rospy.loginfo("Launched node for object detector")
            
        rospy.spin()

    def detect(self, input_image, color_img, depth_img=None):

        # Run inference
        t0 = rospy.get_time()

        img, im0 = input_image, color_img

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        print("input img : ", img.shape)
        print("color_img : ", color_img.shape)
        # Detection
        pred = self.detection_model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t1 = rospy.get_time()
        rospy.loginfo("Detecion inference time : (%.3fs),(%.3f Hz)"%(t1 - t0, 1/(t1 - t0)))
        
        # Detection process
        for det in pred:
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                labels = []

                for *xyxy, conf, cls in det:

                    x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                    if self.publish_image:
                        labels.append(self.names[int(cls)])
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)


                if self.use_depth:
                    self.inference_publish(im0, bbox_xywh, confs, labels, depth_img)
                else:
                    self.inference_publish(im0, bbox_xywh, confs, labels)

        
        t3 = rospy.get_time()
        rospy.loginfo('%sDone. (%.3fs),(%.3f Hz)' % (s, t3 - t0, 1/(t3 - t0)))
       

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
            tmp_box.score = float(score[0])
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

            rospy.loginfo("Check time : (%.3fs),(%.3f Hz)"%(end - start, 1/(end - start)))

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
 
    def image_raw_to_input_image(self, raw_image):

        input_image = letterbox(raw_image, new_shape=self.img_size)[0]

        # Convert
        input_image = input_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        input_image = np.ascontiguousarray(input_image)
        
        return input_image

    def bbox_rel(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        """ xyxy = (x1, y1, x2, y2)  -> (x_C, y_c, w, h)"""
        # print('xyxy : [{}, {}, {}, {}]'.format(xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

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
        
        with torch.no_grad():
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
        
        with torch.no_grad():
            self.detect(input_image, color_image)
        
        rospy.loginfo('inference end!')


def main():

    # Initialize node
    rospy.init_node('detector_node')
    
    # Define traker
    detector_node = DetectorModel()

    # rate = rospy.Rate(3)
    # while not rospy.is_shutdown():
    #     tracking_node.run_tracking()
    #     rate.sleep

if __name__ == '__main__':
    main()
