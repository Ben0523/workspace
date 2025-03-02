#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import pygame

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

        #pygame
        pygame.init()
        self.colors = {}
        self.running = True
        self.screen_width, self.screen_height = 640, 480
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        pygame.display.set_caption("Cone locations")  
        self.scale = 1
        self.offset_x, self.offset_y = 0,0
        self.robot_triangle_coordinates = [(self.screen_width//2-10, self.screen_height),(self.screen_width//2+10, self.screen_height),(self.screen_width//2, self.screen_height-20)]
        #----
        
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

            self.render(center_points)
            
            self.rate.sleep()

    def render(self, dataset):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((255,255,255))
        font = pygame.font.Font(None, 32)
        #pygame.draw.polygon(self.screen, [255,0,0], self.robot_triangle_coordinates)
        #robot_label = font.render("robot", [0,0,0], True)
        #self.screen.blit(robot_label,(int(self.offset_x), int(self.offset_y - 25)))



        for point in dataset:
            x, y = point[0], point[1]
            screen_x, screen_y = self.transform_point(x, y)
            pygame.draw.circle(self.screen,[0,0,0], (screen_x, screen_y), 5)
            
        pygame.display.flip()   

    def transform_point(self,x, y):
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(y * self.scale + self.offset_y)
        return screen_x, screen_y

if __name__ == "__main__":
    try:
        detector = YoloConeDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
