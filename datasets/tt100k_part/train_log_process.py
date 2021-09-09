#!/usr/bin/env python
# encoding: utf-8
 
 
import os
import argparse
import re
import collections
import matplotlib.pyplot as plt
 
 
parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_100/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_250-350/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_350-400/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_0-600/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano_0-100_1e-6/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_tiny_100_1e-6_no_aug/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano3_30_1e-5/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano3_100_5e-5/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano3_200_2e-4/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano5_400_5e-4/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano5_400-500_1e-5/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano5_500-700_5e-6/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano5_700-1000_1e-6/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano640_45_200_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano45t_416_0_0.1_20_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano45t_416_0_0.5_30_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano3_640NP_0_0.5_500_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_nano11_416Pre_0_0.5_400_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_tiny11_416Pre_0_0.5_500_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/test_nano640_210906/yolox_tt100k_nano11_416NP_0_0.5_20_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/test_nano640_210906/yolox_tt100k_nano11_416Pre_0_0.5_20_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/test_nano640_210906/yolox_tt100k_nano11_640NP_newCs_0_0.5_20_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_nt11_640_210906/yolox_tt100k_nano11_640Pre_400_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_nt11_640_210906/yolox_tt100k_nano11_1024Pre_100_1e-3/train_log.txt",
                    default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_nt11_640_210906/yolox_tt100k_nano11_1024Pre_100-300_5e-5/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_nt11_640_210906/yolox_tt100k_tiny11_1024Pre_100_1e-3/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_0-300/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_0-300_r216_b8/train_log.txt",
                    # default="/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/yolox_tt100k_s_100-250/train_log.txt",
                    type=str,
                    help='input file name')
args = parser.parse_args()


def plot_any_x_y(info, x, y, clr='b'):
    """ x & y are string"""
    # i = 0
    for point in info:
        plt.plot(eval('point.'+x), eval('point.'+y), clr+'.', ms=1, label=y)

    plt.xlabel(x)
    plt.ylabel(y)
    return


def moving_average(x: list(), y: list(), avg_count) -> float:
    if len(x) < avg_count + 1:
        return sum(x) / len(x)
    else:
        return (y[-1]*avg_count + x[-1] - x[-1-avg_count]) / avg_count


def main():
    file_name = args.input
    file_path = os.path.dirname(args.input)
    if not os.path.exists(args.input):
        print("Input file not exist!")
        return -1

    """ line demo:
    2021-08-25 17:14:37.032 | INFO     | yolox.core.trainer:after_iter:247 - epoch: 1/100, iter: 10/755, mem: 5578Mb, iter_time: 0.291s, data_time: 0.001s, total_loss: 25.1, iou_loss: 2.4, l1_loss: 0.0, conf_loss: 15.8, cls_loss: 6.9, lr: 2.193e-07, size: 640, ETA: 6:06:13
    2021-08-27 14:15:39.497 | INFO     | yolox.evaluators.tt100k_evaluator:evaluate_prediction:208 - val/mAP50: 0.8431, val/mAP50_95: 0.6321
    """ 
    line_pt = re.compile('(.*)epoch: ([0-9.]+)/([0-9.]+), iter: ([0-9.]+)/([0-9.]+),(.*)total_loss: ([0-9.]+), iou_loss: ([0-9.]+), l1_loss: ([0-9.]+), conf_loss: ([0-9.]+), cls_loss: ([0-9.]+), lr: (.*), size(.*)')
    line_pt_ap = re.compile('(.*)val/mAP50: ([0-9.]+), val/mAP50_95: ([0-9.]+)')

    result_list = list()
    result_dict_no_reap = collections.OrderedDict()
    result_dict_ap_no_reap = collections.OrderedDict()
    result_dict = {"epoch": list(), "iter": list(), "iter_all": list(), "epoch_num": list(),
                   "total_loss": list(), "iou_loss": list(), "l1_loss": list(), "conf_loss": list(), "cls_loss": list(),
                   "total_loss_avg": list(), "iou_loss_avg": list(), "l1_loss_avg": list(), "conf_loss_avg": list(), "cls_loss_avg": list(),
                   "lr": list()}
    result_dict_ap = {"epoch_num": list(), "mAP50": list(), "mAP50_90": list()}
    iter_one_ep = 755   # 503   # 1509  # 1006  # 755
    avg_count = 10
    last_epoch, last_iter = 0.0, 0
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line_ret = line_pt.match(line)
            if line_ret is not None:
                dict_tmp = {"epoch": 0, "iter": 0, "iter_all": 0, "epoch_num": 0.0,
                            "total_loss": 0.0, "iou_loss": 0.0, "l1_loss": 0.0, "conf_loss": 0.0, "cls_loss": 0.0, "lr": 0.0}
                dict_tmp["epoch"] = int(line_ret.group(2))
                dict_tmp["iter"] = int(line_ret.group(4))
                dict_tmp["iter_all"] = (dict_tmp["epoch"]-1)*iter_one_ep + dict_tmp["iter"]
                dict_tmp["epoch_num"] = dict_tmp["epoch"] - 1 + dict_tmp["iter"]/iter_one_ep
                last_epoch, last_iter = dict_tmp["epoch_num"], dict_tmp["iter_all"]
                dict_tmp["total_loss"] = float(line_ret.group(7))
                dict_tmp["iou_loss"] = float(line_ret.group(8))
                dict_tmp["l1_loss"] = float(line_ret.group(9))
                dict_tmp["conf_loss"] = float(line_ret.group(10))
                dict_tmp["cls_loss"] = float(line_ret.group(11))
                dict_tmp["lr"] = float(line_ret.group(12))
                result_list.append(dict_tmp)
                result_dict_no_reap[dict_tmp["iter_all"]] = dict_tmp    # overwrite repeated epoch&iter
            else:
                line_ret_ap = line_pt_ap.match(line)
                if line_ret_ap is not None:
                    dict_tmp = {"epoch_num": last_epoch, "mAP50": float(line_ret_ap.group(2)),
                                "mAP50_90": float(line_ret_ap.group(3))}
                    result_dict_ap_no_reap[last_iter] = dict_tmp

    for _, dict_tmp in result_dict_no_reap.items():
        result_dict["epoch"].append(dict_tmp["epoch"])
        result_dict["iter"].append(dict_tmp["iter"])
        result_dict["iter_all"].append(dict_tmp["iter_all"])
        result_dict["epoch_num"].append(dict_tmp["epoch_num"])
        result_dict["total_loss"].append(dict_tmp["total_loss"])
        result_dict["iou_loss"].append(dict_tmp["iou_loss"])
        result_dict["l1_loss"].append(dict_tmp["l1_loss"])
        result_dict["conf_loss"].append(dict_tmp["conf_loss"])
        result_dict["cls_loss"].append(dict_tmp["cls_loss"])
        result_dict["lr"].append(dict_tmp["lr"])

        if len(result_dict["iter_all"]) > avg_count + 1 and \
           result_dict["iter_all"][-1] < result_dict["iter_all"][-1 - avg_count]:
            print("Sequence error with iter num!")

        # moving average
        result_dict["total_loss_avg"].append(moving_average(result_dict["total_loss"],
                                                            result_dict["total_loss_avg"], avg_count))
        result_dict["iou_loss_avg"].append(moving_average(result_dict["iou_loss"],
                                                          result_dict["iou_loss_avg"], avg_count))
        result_dict["l1_loss_avg"].append(moving_average(result_dict["l1_loss"],
                                                         result_dict["l1_loss_avg"], avg_count))
        result_dict["conf_loss_avg"].append(moving_average(result_dict["conf_loss"],
                                                           result_dict["conf_loss_avg"], avg_count))
        result_dict["cls_loss_avg"].append(moving_average(result_dict["cls_loss"],
                                                          result_dict["cls_loss_avg"], avg_count))

    for _, dict_tmp in result_dict_ap_no_reap.items():
        result_dict_ap["epoch_num"].append(dict_tmp["epoch_num"])
        result_dict_ap["mAP50"].append(dict_tmp["mAP50"])
        result_dict_ap["mAP50_90"].append(dict_tmp["mAP50_90"])

    plt_conf = "MULTI"
    if plt_conf == 'MULTI':
        plt.figure(22, figsize=(16, 9), dpi=80)
        ax = plt.subplot(221)
        ax.set_ylim(0, 10)
        ax.grid()
        plt.title("loss")
        plt.xlabel("epoch")
        plt.plot(result_dict["epoch_num"], result_dict["total_loss"], label="total_loss")
        plt.plot(result_dict["epoch_num"], result_dict["iou_loss"], label="iou_loss")
        plt.plot(result_dict["epoch_num"], result_dict["l1_loss"], label="l1_loss")
        plt.plot(result_dict["epoch_num"], result_dict["conf_loss"], label="conf_loss")
        plt.plot(result_dict["epoch_num"], result_dict["cls_loss"], label="cls_loss")
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

        ax = plt.subplot(222)
        ax.set_ylim(0, 10)
        ax.grid()
        plt.title("loss_avg")
        plt.xlabel("epoch")
        plt.plot(result_dict["epoch_num"], result_dict["total_loss_avg"], label="total_loss_avg")
        plt.plot(result_dict["epoch_num"], result_dict["iou_loss_avg"], label="iou_loss_avg")
        plt.plot(result_dict["epoch_num"], result_dict["l1_loss_avg"], label="l1_loss_avg")
        plt.plot(result_dict["epoch_num"], result_dict["conf_loss_avg"], label="conf_loss_avg")
        plt.plot(result_dict["epoch_num"], result_dict["cls_loss_avg"], label="cls_loss_avg")
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

        ax = plt.subplot(223)
        ax.grid()
        plt.title("lr")
        plt.xlabel("iters")
        plt.plot(result_dict["epoch_num"], result_dict["lr"], label="lr")

        ax = plt.subplot(224)
        print(result_dict_ap)
        # ax.set_ylim(0.0, 1)
        ax.grid()
        plt.title("mAP")
        plt.xlabel("epoch")
        plt.plot(result_dict_ap["epoch_num"], result_dict_ap["mAP50"], "*", label="mAP50")
        plt.plot(result_dict_ap["epoch_num"], result_dict_ap["mAP50_90"], "*", label="mAP50_90")
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    else:
        plt.figure(figsize=(16, 9), dpi=80)
        ax = plt.subplot()
        ax.set_ylim(0, 20)
        ax.grid()
        plt.title("loss")
        plt.xlabel("epoch")
        plt.plot(result_dict["epoch_num"], result_dict["total_loss"], label="total_loss")
        plt.plot(result_dict["epoch_num"], result_dict["iou_loss"], label="iou_loss")
        plt.plot(result_dict["epoch_num"], result_dict["l1_loss"], label="l1_loss")
        plt.plot(result_dict["epoch_num"], result_dict["conf_loss"], label="conf_loss")
        plt.plot(result_dict["epoch_num"], result_dict["cls_loss"], label="cls_loss")
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')


    loss_fig = os.path.join(file_path, "train_loss.jpg")
    plt.savefig(loss_fig)
    plt.show()
 
 
if __name__ == '__main__':
    main()