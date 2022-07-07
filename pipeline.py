import cv2
import time 
import os 

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from libs import CRAFT_DETECTOR
from utils.create_output_file import create_output
from configs.config import init_config

# config
CFG = init_config()
DEVICE = CFG["device"]["device"]
craft_weight_path = CFG["craft"]["weight_path"]
vietocr_weight_path = CFG["vietocr"]["weight_path"]

# TEXT_DETECTOR
TEXT_DETECTOR = CRAFT_DETECTOR(craft_weight_path, DEVICE)

# TEXT RECOGNITION
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = vietocr_weight_path
config['cnn']['pretrained']=False
# config['device'] = 'cuda:0' 
config['device'] = CFG["device"]["device"]
config['predictor']['beamsearch']=False

TEXT_RECOGNIZER= Predictor(config)

def predict_image(image_path, result_dir="dataset/results"):
    image_name = (image_path.split("/")[-1]).split(".")[0]
    image = cv2.imread(image_path)
    image_visual = image.copy()

    time_start =  time.time()
    # detect 
    horizontal_list, free_list = TEXT_DETECTOR.readtext(image)

    for bbox in horizontal_list:
        crop_image = TEXT_DETECTOR.crop_image(image, bbox)

        result_text = TEXT_RECOGNIZER.predict(crop_image)

        # visualize 
        # result_dir = "dataset/result_images"
        # visual_image(image, bbox, result_text, result_dir)
        # visual_image_path = os.path.join(result_dir, image_name + ".jpg")
        

        # visualize
        image_visual = TEXT_DETECTOR.visualize_box_text(image_visual, bbox, result_text)

    result_dir = "dataset/result_images"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    visual_image_path = os.path.join(result_dir, image_name + ".jpg")
    cv2.imshow(visual_image_path, image_visual)

    print("Time process: ", time.time() - time_start)

if __name__ == "__main__":
    image_path = "dataset/images/demo.png"
    predict_image(image_path)