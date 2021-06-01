import cv2
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = 'models/vietocr/transformerocr.pth'
config['cnn']['pretrained']=False
# config['device'] = 'cuda:0' 
config['device'] = 'cpu' 
config['predictor']['beamsearch']=False

detector = Predictor(config)

# image = cv2.imread("crop_image.jpg")
image = img = Image.open("crop_image.jpg")

result_text = detector.predict(image)
print(result_text)