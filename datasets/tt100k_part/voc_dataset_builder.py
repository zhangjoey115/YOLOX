import os
import subprocess
import re


script_path = os.path.realpath(__file__)
script_path = os.path.dirname(script_path)


def process_pic_batch_voc(dataset_files, dataset_dir, anno_dir, pic_num_max, id_file_name="train"):
    # move N pic&xml to form VOC dataset
    i = 0
    pic_files = []
    for name in dataset_files:
        file_name = name.split('.')[0]
        cmd = "cp " + anno_dir + file_name + ".xml " + " tt100k2021/Annotations/"
        ret = subprocess.call(cmd,  shell=True)
        if ret != 0:
            print('xml not find, skip file {}'.format(file_name))
            continue

        cmd = "cp " + dataset_dir + name + " " + " tt100k2021/JPEGImages/"
        subprocess.call(cmd,  shell=True)

        pic_files.append(file_name)

        i += 1
        if i >= pic_num_max:
            break
    with open(id_file_name, 'w') as f:
        for name in pic_files:
            f.write(name + '\r')


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
    top_num = 45
    top_list = list()
    for i in range(top_num):
        top_list.append(class_dict_sort_v[i][0])
    print("Top {} classes are: ".format(top_num))
    top_list_sort = sorted(top_list)
    print(top_list_sort)

    print("Top {} classes for TT100K_CLASSES is: ".format(top_num))
    for name in top_list_sort:
        print("\t\"{}\",".format(name))


if __name__ == "__main__":
    # option = 'BUILD_DATASET'
    option = 'ANALYSE_DATASET'
    if option == 'BUILD_DATASET':
        subprocess.call("rm -rf tt100k2021",  shell=True)
        subprocess.call("mkdir -p tt100k2021/Annotations",  shell=True)
        subprocess.call("mkdir -p tt100k2021/JPEGImages",  shell=True)
        subprocess.call("mkdir -p tt100k2021/ImageSets/Main",  shell=True)

        # train pics
        pic_num_train = 10000
        # dir should end with /
        dataset_dir_train = "/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/train/"
        dataset_files_train = os.listdir(dataset_dir_train)
        anno_dir_train = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/train/"
        id_file_name = "tt100k2021/ImageSets/Main/train.txt"

        process_pic_batch_voc(dataset_files_train, dataset_dir_train, anno_dir_train, pic_num_train, id_file_name)

        # test pics
        pic_num_test = 10000
        dataset_dir_test = "/home/zjw/workspace/DL_Vision/dataset/TT100k/tt100k_2021/test/"
        dataset_files_test = os.listdir(dataset_dir_test)
        anno_dir_test = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/test/"
        id_file_name = "tt100k2021/ImageSets/Main/test.txt"

        process_pic_batch_voc(dataset_files_test, dataset_dir_test, anno_dir_test, pic_num_test, id_file_name)

    elif option == 'ANALYSE_DATASET':
        print("Train Summary:")
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/train/"
        analyse_dataset(anno_dir)
        print("Test Summary:")
        anno_dir = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/datasets/tt100k_part/xmlLabel/test/"
        analyse_dataset(anno_dir)
