import os 
import numpy as np 
import cv2
from PIL import Image, ImageDraw, ImageFont

def create_output(image_name, bbox, result_text, result_dir):
    x_min = bbox[0]
    x_max = bbox[1]
    y_min = bbox[2]
    y_max = bbox[3]

    # create format: tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y, text
    content = "{},{},{},{},{},{},{},{},{}\n".format(
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, result_text)
    
    file_path = os.path.join(result_dir, image_name +'.txt')

    with open(file_path, "a+") as f:
        f.write(content)

def write_text(image, result_text, point, **option):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print("result_text: ", result_text)
    font_text = ImageFont.truetype("models/Roboto-Medium.ttf", 16)
    draw = ImageDraw.Draw(pil_img)
    draw.text(point, result_text, (0, 255, 255), font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    
    return cv2_img

def draw_text_bbox(image, list_bboxes, list_result_text, color=(255, 0, 0), thickness=2, isClosed=True):
    for bbox, result_text in zip(list_bboxes, list_result_text):
        pts = np.array(bbox)
        image = cv2.polylines(image, [pts], isClosed, color, thickness)
        image = write_text(image, result_text, (bbox[0][0], bbox[0][1]))

    return 