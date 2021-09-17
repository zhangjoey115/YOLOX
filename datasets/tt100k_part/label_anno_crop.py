import os
import re
import json
from lxml import etree as ET
from xml.dom import minidom


def edit_xml(objects, file_name, save_root, wid=100, hi=100):
    """
    object = {'category': 'pl5', 'bbox': {'xmin': 0, 'xmax': 100, 'ymin': 0, 'ymax': 100}}
    """
    save_xml_path = os.path.join(save_root, "%s.xml" % file_name)  # xml

    root = ET.Element("annotation")
    # root.set("version", "1.0")  
    folder = ET.SubElement(root, "folder")
    folder.text = "none"
    filename = ET.SubElement(root, "filename")
    filename.text = file_name + ".jpg"
    source = ET.SubElement(root, "source")
    source.text = "none"
    owner = ET.SubElement(root, "owner")
    owner.text = "zjw"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(wid)
    height = ET.SubElement(size, "height")
    height.text = str(hi)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    for obj in objects:  #  
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")  # number
        name.text = obj["category"]
        # meaning = ET.SubElement(object, "meaning")  # name
        # meaning.text = inf_value[0]
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(obj["bbox"]["xmin"]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(obj["bbox"]["ymin"]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(obj["bbox"]["xmax"]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(obj["bbox"]["ymax"]))
    tree = ET.ElementTree(root)
    tree.write(save_xml_path, encoding="UTF-8", xml_declaration=True)
    root = ET.parse(save_xml_path) 
    file_lines = minidom.parseString(ET.tostring(root, encoding="Utf-8")).toprettyxml(
        indent="\t") 
    file_line = open(save_xml_path, "w", encoding="utf-8")
    file_line.write(file_lines)
    file_line.close()


def get_img_dir_anno(dir):
    # get the  id list  of xx_xx_id.jpg
    names = os.listdir(dir)
    file_names = []
    type_names = []
    for name in names:
        # name may be '38741_2_ph3.2.jpg' 
        file_name = name[:-4]
        file_names.append(file_name)
        type_names.append(re.split("_", file_name)[-1])
    return file_names, type_names  


if __name__ == "__main__":
    # img_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/train/'
    # anno_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/anno_train/'
    img_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/test/'
    anno_dir = '/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/anno_test/'
    os.makedirs(anno_dir, exist_ok=True)

    file_names, type_names = get_img_dir_anno(img_dir)

    for i, (file_name, type_name) in enumerate(zip(file_names, type_names)):
        objects = [{'category': type_name, 'bbox': {'xmin': 0, 'xmax': 100, 'ymin': 0, 'ymax': 100}}]
        edit_xml(objects, file_name, anno_dir)

        if (i+1) % 100 == 0:
            print("Finish generate {} annos".format(i+1))
