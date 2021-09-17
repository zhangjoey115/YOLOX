import os
import subprocess
import re


script_path = os.path.realpath(__file__)
script_path = os.path.dirname(script_path)


def process_pic_batch_voc(dataset_files, dataset_dir, anno_dir, pic_num_max,
                          dataset_name_new='tt100k2021', id_file_name="train"):
    # move N pic&xml to form VOC dataset
    i = 0
    pic_files = []
    for name in dataset_files:
        file_name = name[:-4]
        cmd = "cp " + anno_dir + file_name + ".xml " + dataset_name_new + "/Annotations/"
        ret = subprocess.call(cmd,  shell=True)
        if ret != 0:
            print('xml not find, skip file {}'.format(file_name))
            continue

        cmd = "cp " + dataset_dir + name + " " + dataset_name_new + "/JPEGImages/"
        subprocess.call(cmd,  shell=True)

        pic_files.append(file_name)

        i += 1
        if i >= pic_num_max:
            break
    with open(id_file_name, 'w') as f:
        for name in pic_files:
            f.write(name + '\r')
    print("Finish process dataset " + id_file_name)


def analyse_dataset(anno_dir):
    class_dict = dict()
    anno_list = os.listdir(anno_dir)
    for anno_file in anno_list:
        with open(os.path.join(anno_dir, anno_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find("<name>") != -1:
                    name = re.split("<|>", line)[2]
                    if name in class_dict.keys():
                        class_dict[name] += 1
                    else:
                        class_dict[name] = 1
    total_num = total_cls = 0
    for v in class_dict.items():
        total_cls += 1
        total_num += v[1]
    print("  SignTypeNum = {}, SignTotalNum = {}".format(total_cls, total_num))
    class_dict_sort_k = sorted(class_dict.items(), key=lambda item: item[0], reverse=False)
    class_dict_sort_v = sorted(class_dict.items(), key=lambda item: item[1], reverse=True)
    print(class_dict_sort_k)
    print(class_dict_sort_v)

    # find top several classes
    top_num = 150   # 45
    top_list = list()
    for i in range(top_num):
        top_list.append(class_dict_sort_v[i][0])
    print("Top {} classes are: ".format(top_num))
    top_list_sort = sorted(top_list)
    print(top_list_sort)

    print("Top {} classes for TT100K_CLASSES is: ".format(top_num))
    for name in top_list_sort:
        print("\t\"{}\",".format(name))

    # print i/p/w count
    cls_cnt = {'i': 0, 'p': 0, 'w': 0}
    for v in class_dict.items():
        name = v[0]
        if name.startswith('i'):
            cls_cnt['i'] += v[1]
        elif name.startswith('p'):
            cls_cnt['p'] += v[1]
        elif name.startswith('w'):
            cls_cnt['w'] += v[1]
        else:
            print("Error with unknown class {}!".format(name))
    print('Class count i = {}, p = {}, w = {}.'.format(cls_cnt['i'], cls_cnt['p'], cls_cnt['w']))


def divide_anno_classes(name, cls_cnt, cls_cnt_sub):
    """
    class_div0 = {'45classes'}
    class_div1 = {'45classes', 'other'}
    class_div = {'i': ['il', 'ip', 'i_other'],
                 'p': ['pm', 'pa', 'pl', 'pr', 'ph', 'pw', 'p_other'],
                 'w': ['w']}
    class_div2 = {'i', 'p', 'w'}
    class_div3 = {'i_num', 'ip', 'i_other'
                 'p_num', 'p_other'
                 'w'}
    class_div4 = {'i_num', 'ip', 'i_other'
                 'p_num', 'pn_x', 'ps', 'pg', 'p_other'
                 'w'}
    class_div = {'i': ['il', 'ip', 'i_other'],                          # 11
                 'p': ['pm', 'pa', 'pl', 'pr', 'ph', 'pw', 'p_other'],
                 'w': ['w']}
    class_div = {'i': ['il50', 'il60', 'il70', 'il80', 'il90', 'il100', 'il110',
                       'ip', 'i_other'],
                 'p': ['pm20', 'pm30', 'pm55', xxx
                       'pa10', 'pa12', 'pa13', 'pa14',
                       'pl5', 'pl10', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl50', 'pl60', 'pl70', 'pl80', 'pl90', 'pl100', 'pl110', 'pl120',
                       'pr', xxx
                       'ph1.8', 'ph2', 'ph2.2', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.5', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5',
                       'pw', xxx
                       'p_other'],
                 'w': ['w']}
    class_div = {'i': ['other'],                          # only pl
                 'pl': ['pl5', 'pl15', 'pl20', 'pl25', 'pl30', 'pl40', 'pl50', 'pl60', 'pl70', 'pl80', 'pl90', 'pl100', 'pl110', 'pl120', 'pl_other'],
                 'p': ['other'],
                 'w': ['other']}
    """
    # cls_cnt = {'i': 0, 'p': 0, 'w': 0}
    # class_div = {'i': ['il', 'ip', 'i_other'],
    #              'p': ['pm', 'pa', 'pl', 'pr', 'ph', 'pw', 'p_other'],
    #              'w': ['w']}
    class_div = {'i': ['other'],                          # only pl
                 'pl': ['pl5', 'pl15', 'pl20', 'pl25', 'pl30', 'pl40', 'pl50', 'pl60', 'pl70', 'pl80', 'pl90', 'pl100', 'pl110', 'pl120', 'pl_other'],
                 'p': ['other'],
                 'w': ['other']}
    is_full_match = True

    def divide_sub_anno(name, class_list, full_match=False):
        name_new = ''
        for type in class_list:
            if not full_match and name.startswith(type):
                name_new = type
                break
            elif full_match and name == type:
                name_new = type
                break
        if name_new == '':
            name_new = class_list[-1]
        return name_new

    for k in class_div.keys():
        if name.startswith(k):
            cls_cnt[k] = 1 if k not in cls_cnt.keys() else cls_cnt[k]+1
            name_new = divide_sub_anno(name, class_div[k], full_match=is_full_match)
            break
    cls_cnt_sub[name_new] = 1 if name_new not in cls_cnt_sub.keys() else cls_cnt_sub[name_new] + 1
    # if name.startswith('i'):
    #     cls_cnt['i'] += 1
    #     name_new = divide_sub_anno(name, class_div['i'])
    # elif name.startswith('p'):
    #     cls_cnt['p'] += 1
    #     name_new = divide_sub_anno(name, class_div['p'])
    # elif name.startswith('w'):
    #     cls_cnt['w'] += 1
    #     name_new = divide_sub_anno(name, class_div['w'])
    # else:
    #     print("Error with unknown class {}!".format(name))

    return name_new, cls_cnt, cls_cnt_sub


def change_annotation(anno_dir, anno_dir_new=None):
    if not os.path.exists(anno_dir):
        print("Annotation Dir not Exist!")
        return

    os.makedirs(anno_dir_new, exist_ok=True)
    anno_list = os.listdir(anno_dir)
    # cls_cnt = {'i': 0, 'p': 0, 'w': 0}
    cls_cnt = {}
    cls_cnt_sub = {}
    for anno_file in anno_list:
        lines = []
        with open(os.path.join(anno_dir, anno_file), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(anno_dir_new, anno_file), 'w') as f:
            for line in lines:
                if line.find("<name>") != -1:
                    words = re.split("<|>", line)
                    name = words[2]
                    name, cls_cnt, cls_cnt_sub = divide_anno_classes(name, cls_cnt, cls_cnt_sub)
                    line_new = line.replace(words[2], name, 1)
                    # line_new = '<name>' + name + '</name>'
                    f.write(line_new)                    
                else:
                    f.write(line)
    # print('Finish Anno Change! Class count i = {}, p = {}, w = {}.'.format(cls_cnt['i'], cls_cnt['p'], cls_cnt['w']))
    print(cls_cnt)
    print(cls_cnt_sub)


if __name__ == "__main__":
    # option = 'BUILD_DATASET'
    # option = 'ANALYSE_DATASET'
    # option = 'CHANGE_ANNOTATION'
    option = 'CHANGE_ANNOTATION'
    if option == 'BUILD_DATASET':
        # dir should end with /
        # dataset_name = "tt100k2021"     # "tt100k_crop_100_16"
        # dataset_dir_train = "/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/train/"
        # anno_dir_train = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/train/"
        # id_file_name_train = "tt100k2021/ImageSets/Main/train.txt"
        # dataset_dir_test = "/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/test/"
        # anno_dir_test = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/test/"
        # id_file_name_test = "tt100k2021/ImageSets/Main/test.txt"

        dataset_name = "tt100k_crop_100_16"     # "tt100k2021"
        dataset_dir_train = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/train/"
        anno_dir_train = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/anno_train/"
        id_file_name_train = dataset_name + "/ImageSets/Main/train.txt"
        dataset_dir_test = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/test/"
        anno_dir_test = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021_crop/anno_test/"
        id_file_name_test = dataset_name + "/ImageSets/Main/test.txt"

        subprocess.call("rm -rf " + dataset_name,  shell=True)
        subprocess.call("mkdir -p " + dataset_name + "/Annotations",  shell=True)
        subprocess.call("mkdir -p "+ dataset_name + "/JPEGImages",  shell=True)
        subprocess.call("mkdir -p " + dataset_name +"/ImageSets/Main",  shell=True)

        # train pics
        pic_num_train = 1e7
        dataset_files_train = os.listdir(dataset_dir_train)
        process_pic_batch_voc(dataset_files_train, dataset_dir_train, anno_dir_train, pic_num_train, dataset_name, id_file_name_train)

        # test pics
        pic_num_test = 1e7
        dataset_files_test = os.listdir(dataset_dir_test)
        process_pic_batch_voc(dataset_files_test, dataset_dir_test, anno_dir_test, pic_num_test, dataset_name, id_file_name_test)

    elif option == 'ANALYSE_DATASET':
        print("Train Summary:")
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/train/"
        analyse_dataset(anno_dir)
        print("Test Summary:")
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/test/"
        analyse_dataset(anno_dir)

    elif option == 'CHANGE_ANNOTATION':
        print("Change Annotation to 3 classed: i/p/w")
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021/Annotations_org/"
        anno_dir_new = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k2021/Annotations_pl/"
        # anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k_crop_100_16/Annotations_org/"
        # anno_dir_new = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/tt100k_crop_100_16/Annotations_pl/"
        change_annotation(anno_dir, anno_dir_new)
