import os 

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