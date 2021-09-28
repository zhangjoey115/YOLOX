"""
    Info:   Visualize pic with xml formate label. 
    
    Usage:  
    Author: zjw
    Date:   2021-09-27
"""
import os
import cv2
import subprocess
import xml.etree.ElementTree as ET


script_path = os.path.realpath(__file__)
script_path = os.path.dirname(script_path)


def create_one_label_image(pic_path, labels):
    img = cv2.imread(pic_path)

    for i in range(len(labels)):
        box = labels[i]['bbox']
        name = labels[i]['name']
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(name, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.rectangle(
            img,
            (x0, y0 - int(2*txt_size[1])),
            (x0 + txt_size[0] + 1, y0 - 1),
            (0, 255, 0),
            -1
        )
        cv2.putText(img, name, (x0, y0 - txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)
    # cv2.imshow(name, img)
    return img


def parse_anno_label(anno_path):
    labels = []
    try:
        target = ET.parse(anno_path).getroot()
    except Exception:
        print("Error while parse anno file: " + anno_path)
        return None

    for obj in target.iter("object"):
        difficult = obj.find("difficult")
        if difficult is not None:
            difficult = int(difficult.text) == 1
        else:
            difficult = False

        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")

        pts = ["xmin", "ymin", "xmax", "ymax"]
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(round(float(bbox.find(pt).text)))
            bndbox.append(cur_pt)
        label = {'name': name, 'bbox': bndbox}
        labels.append(label)

    return labels


def create_label_images(anno_dir, pic_dir, pic_dir_new=None):
    if not os.path.exists(anno_dir):
        print("Annotation Dir not Exist!")
        return
    elif not os.path.exists(pic_dir):
        print("Pic Dir not Exist!")
        return
    if pic_dir_new is None:
        suffix = '_label'
        pic_dir_new = pic_dir + suffix
        print("Pic output dir is " + pic_dir_new)
    os.makedirs(pic_dir_new, exist_ok=True)

    anno_list = os.listdir(anno_dir)
    for anno_file in anno_list:
        print("Process image with label " + anno_file)
        anno_path = os.path.join(anno_dir, anno_file)
        labels = parse_anno_label(anno_path)
        if labels is None:
            continue

        pic_path = os.path.join(pic_dir, anno_file[:-4]+'.jpg')
        result_img = create_one_label_image(pic_path, labels)

        pic_path_new = os.path.join(pic_dir_new, anno_file[:-4]+'.jpg')
        cv2.imwrite(pic_path_new, result_img)
        

if __name__ == "__main__":
    # option = 'FILTER_PIC_FILES'
    option = 'FILTER_PIC_FILES'
    if option == 'FILTER_PIC_FILES':
        anno_dir = "/home/zjw/workspace/DL_Vision/dataset/TSR_Label/tsr_label_test_100/XML_DE"
        pic_dir = "/home/zjw/workspace/DL_Vision/dataset/TSR_Label/tsr_label_test_100/JPEG_DE"
        pic_dir_new = None
        create_label_images(anno_dir, pic_dir, pic_dir_new)
        print("Finish generate pics with anno labels!")

