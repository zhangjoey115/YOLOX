"""
    Info:   Chose selected pics by anno type. 
    
    Usage:  FILTER_PIC_FILES:   chose pic like contain pl90.
    Author: zjw
    Date:   2021-09-15
"""
import os
import subprocess
import re


script_path = os.path.realpath(__file__)
script_path = os.path.dirname(script_path)


def check_name_satisfy(name, name_filter, full_match=True):
    """ name_filter = {'pl90': 0} """
    is_match = False
    for k in name_filter.keys():
        if (full_match and name == k) or (not full_match and name.find(k) != -1):
            name_filter[k] = 1 if k not in name_filter.keys() else name_filter[k]+1
            is_match = True

    return is_match


def filter_chosen_pic_from_anno(anno_dir, pic_dir, pic_dir_new=None):
    """
    name_filter = {'pl90': 0}
    name_filter = {'pl90': 0, 'pl110'}
    """
    name_filter = {'pl90': 0}
    FULL_MATCH = True

    if not os.path.exists(anno_dir):
        print("Annotation Dir not Exist!")
        return
    elif not os.path.exists(pic_dir):
        print("Pic Dir not Exist!")
        return
    if pic_dir_new is None:
        suffix = ''
        for k in name_filter.keys():
            suffix += '_' + k
        pic_dir_new = pic_dir + suffix
        print("Pic output dir is " + pic_dir_new)
    os.makedirs(pic_dir_new, exist_ok=True)

    anno_list = os.listdir(anno_dir)
    anno_file_match = []
    for anno_file in anno_list:
        is_match = False
        with open(os.path.join(anno_dir, anno_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find("<name>") != -1:
                    words = re.split("<|>", line)
                    name = words[2]
                    ret = check_name_satisfy(name, name_filter, full_match=FULL_MATCH)
                    is_match |= ret
            if is_match:
                anno_file_match.append(anno_file)
    print(name_filter)
    print("Matched file num is {}".format(len(anno_file_match)))

    cmd = 'cp '
    for anno_name in anno_file_match:
        pic_name = anno_name[:-4] + '.jpg'
        pic_path = os.path.join(pic_dir, pic_name)
        if os.path.exists(pic_path):
            cmd += pic_path + ' '
    cmd += pic_dir_new
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    # option = 'FILTER_PIC_FILES'
    option = 'FILTER_PIC_FILES'
    if option == 'FILTER_PIC_FILES':
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021/Annotations"
        pic_dir = "/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/train"
        pic_dir_new = None
        filter_chosen_pic_from_anno(anno_dir, pic_dir, pic_dir_new)
        print("Finish filter pic files by anno name!")
