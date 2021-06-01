from libs import CRAFT_DETECTOR
import cv2

TEXT_DETECTOR = CRAFT_DETECTOR() 

image = cv2.imread("images/demo.png")
horizontal_list, free_list = TEXT_DETECTOR.readtext(image)

image_drawed = TEXT_DETECTOR.visualize_bbox(image, horizontal_list, (0, 255, 255))

cv2.imshow("image", image)
cv2.waitKey(0)