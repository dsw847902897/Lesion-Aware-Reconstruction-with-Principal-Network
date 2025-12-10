#!/usr/bin/python

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/home/wy/py_doc/GII/generative_inpainting/data/health', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='/home/wy/py_doc/GII/generative_inpainting/data/train.flist', type=str,
                    help='The output filename.')


if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories
    dirs = os.listdir(args.folder_path)
    dirs_name_list = []

    # make 2 lists to save file paths
    training_file_names = []

    # print all directory names
    for dir_item in dirs:
        # modify to full path -> directory
        dir_item = args.folder_path + "/" + dir_item
        # print(dir_item)
        training_file_names.append(dir_item)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename)


