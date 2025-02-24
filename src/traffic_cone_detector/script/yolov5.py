#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch


class YoloConeDetector:
    def __init__(self, camera_port=0):
        rospy.init_node("yolo_cone_detector", anonymous=True)
        self.bridge = CvBridge()
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=10)
        self.model = torch.hub.load("/home/bence/Repos/yolov5", "custom", path="/workspace/src/traffic_cone_detector/script/best.pt", source="local")
        rospy.loginfo("Model loaded successfully!")
        #rospy.loginfo(f"Total model parameters: {sum(p.numel() for p in self.model.model.parameters())}")
        
        self.cap = cv2.VideoCapture(camera_port)
        self.rate = rospy.Rate(30)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to capture image from webcam")
                continue
            
            det_result = self.model(frame, size=384)
            rospy.loginfo("Eredmenyek:")
            rospy.loginfo(det_result)
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        detector = YoloConeDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
