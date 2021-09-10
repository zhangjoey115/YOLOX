import os
import xml.etree.ElementTree as ET
from PIL import Image
import threading

abs_path = os.path.abspath(__file__)
script_path = os.path.realpath(__file__)
script_path = os.path.dirname(script_path)


def create_crop_dataset(img_dir, anno_dir, save_dir):
    img_list = os.listdir(img_dir)
    print("Total image num is {}".format(len(img_list)))
    anno_list = os.listdir(anno_dir)
    # assert len(img_list) == len(anno_list)
    anno_names = [anno.split('.')[0] for anno in anno_list]
    for i, name in enumerate(img_list):
        name_base = name.split('.')[0]
        if name_base not in anno_names:
            continue

        anno_path = os.path.join(anno_dir, name_base+'.xml')
        label_list, _ = get_label_info(anno_path)
        img_path = os.path.join(img_dir, name)
        crop_one_img_by_label(img_path, label_list, save_dir)

        if (i+1) % 100 == 0:
            print("Finish crop {} images".format(i+1))
    pass


def get_label_info(anno_path):
    target = ET.parse(anno_path).getroot()

    label_list = list()
    for obj in target.iter("object"):
        difficult = obj.find("difficult")
        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        pts = ["xmin", "ymin", "xmax", "ymax"]
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)    # - 1
            bndbox.append(cur_pt)
        
        label_list.append({'name': name, 'bbox': bndbox})

    width = int(target.find("size").find("width").text)
    height = int(target.find("size").find("height").text)
    img_size = (height, width)

    return label_list, img_size


def crop_one_img_by_label(img_path, label_list, save_dir):
    """ 
    label_info = list({'name': str, 'bbox': ["xmin", "ymin", "xmax", "ymax"]})
    bbox = (left, upper, right, lower)
    """
    img = Image.open(img_path)
    img_name = os.path.basename(img_path)
    img_prefix = img_name[:-4]
    for i, label in enumerate(label_list):
        out_path = img_prefix + '_' + str(i) + '_' + label['name'] + '.jpg'
        out_path = os.path.join(save_dir, out_path)
        bbox = label["bbox"]
        sub_thrd = threading.Thread(target=crop_save_one_img(img, bbox, out_path))
        sub_thrd.start()
    # print('Finish crop' + img_path)


def crop_save_one_img(img, bbox, out_path, ignore_min=16, resize_wid=100):
    high = bbox[2] - bbox[0]
    wid = bbox[3] - bbox[1]
    if ignore_min > 0 and (high < ignore_min or wid < ignore_min):
        print("Ignore too samll image {} with size = {}, {}".format(os.path.basename(out_path), high, wid))
        return

    cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    if resize_wid > 0 and resize_wid > ignore_min:
        cropped = cropped.resize((resize_wid, resize_wid))
    # cropped.resize()
    # cropped.show()
    cropped.save(out_path)


if __name__ == "__main__":
    # option = 'CROP_SIGNS'
    option = 'CROP_SIGNS'
    if option == 'CROP_SIGNS':
        img_dir = '/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/train/'
        anno_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/train/'
        save_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/JPEGImages2/'
        os.makedirs(save_dir, exist_ok=True)
        create_crop_dataset(img_dir, anno_dir, save_dir)
