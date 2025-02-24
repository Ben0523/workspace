#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

bridge = CvBridge()

class Yolo_cone_detector:
    def __init__(self):
        self.sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback,queue_size=10)
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=10)
        self.model = YOLO('/workspace/src/traffic_cone_detector/script/Cone.pt')
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        rospy.loginfo("Model completed!")
        rospy.loginfo(sum(p.numel() for p in self.model.model.parameters()))
        
    
    def callback(self, msg:Image):
        center_points= []
        
        array = ros_numpy.numpify(msg)

        if self.det_image_pub.get_num_connections():
            det_result = self.model(array, conf=self.confidence_threshold, imgsz=384)

            for r in det_result:
                #rospy.loginfo(r.boxes)
                if r.boxes.xywh.numel() == 0:
                    rospy.loginfo("No object detected!")
                    continue
            
                for box in r.boxes.xywh:
                    x, y, _, _ = box.tolist()
                    x = round(x)
                    y = round(y)
                    center_points.append((x,y))

            rospy.loginfo(center_points)
            det_annotated = det_result[0].plot(show=False)
            self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="bgr8"))
            rospy.loginfo("Publishing...")

        

if __name__ == '__main__':
    rospy.init_node("cone_detector")

    detector_class = Yolo_cone_detector()
    
    rospy.spin()
