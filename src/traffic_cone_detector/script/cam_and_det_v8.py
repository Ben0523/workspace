#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloConeDetector:
    def __init__(self, camera_port=0):
        rospy.init_node("yolo_cone_detector", anonymous=True)
        self.bridge = CvBridge()
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=10)
        self.model = YOLO('/workspace/src/traffic_cone_detector/script/Cone.pt')
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.imgsz = rospy.get_param("~image_size", 480)
        rospy.loginfo("Model loaded successfully!")
        rospy.loginfo(f"Total model parameters: {sum(p.numel() for p in self.model.model.parameters())}")
        
        self.cap = cv2.VideoCapture(camera_port)
        self.rate = rospy.Rate(30)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to capture image from webcam")
                continue
            
            det_result = self.model(frame, conf=self.confidence_threshold, imgsz=self.imgsz)
            center_points = []
            
            for r in det_result:
                if r.boxes.xywh.numel() == 0:
                    rospy.loginfo("No object detected!")
                    continue
                
                for box in r.boxes.xywh:
                    x, y, _, _ = box.tolist()
                    x, y = round(x), round(y)
                    center_points.append((x, y))
            
            rospy.loginfo(center_points)
            det_annotated = det_result[0].plot(show=False)
            
            if self.det_image_pub.get_num_connections():
                self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(det_annotated, encoding='bgr8'))
                rospy.loginfo("Publishing detection result...")
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        detector = YoloConeDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
