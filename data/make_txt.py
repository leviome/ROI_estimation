import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default='./')
    args = parser.parse_args()
    return args


dataset_folder = ''
classes = ["focus"]


def convert_xml(file_path, out_file):
    out_file = open(out_file, 'w')
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        bb = (max(1.0, float(xmlbox.find('xmin').text)),
              max(1.0, float(xmlbox.find('ymin').text)),
              min(float(w - 1), float(xmlbox.find('xmax').text)),
              min(float(h - 1), float(xmlbox.find('ymax').text)))

        out_file.write(",".join([str(a) for a in bb]) + ',' + str(cls_id) + '\n')

    out_file.close()


def _run(clear_txt=True):
    if clear_txt:
        if osp.exists('train.txt'):
            os.system('rm train.txt')
        if osp.exists('test.txt'):
            os.system('rm test.txt')
    else:
        if osp.exists('train.txt') or osp.exists('test.txt'):
            return
    args = parse_args()
    root_dir = ''
    jpg_folder = osp.join(root_dir, dataset_folder, 'JPEGImages')
    name_list = os.listdir(jpg_folder)
    print(len(name_list))
    name_list = tqdm(name_list)
    train_list = open('train.txt', 'w+')
    test_list = open('test.txt', 'w+')
    for it, img_name in enumerate(name_list):
        name = img_name.strip('.jpg')
        label_file = osp.join(root_dir, dataset_folder, 'Label', name + '.txt')
        f = open(label_file, 'r')
        line = f.readlines()[0].strip().split(',')[0:2]
        f.close()
        label_line = ','.join(line)
        img_file = osp.join(jpg_folder, img_name)
        line_content = img_file + ' ' + label_line + '\n'
        if osp.exists(label_file):
            if it < len(name_list) * 2 / 3:
                train_list.write(line_content)
            else:
                test_list.write(line_content)

    train_list.close()
    test_list.close()


def parse_line(line):
    pairs = line.strip().split(' ')
    x = pairs[1].split(',')[0]
    y = pairs[1].split(',')[1]
    return cv2.imread(pairs[0]), [float(x), float(y)]


def check_data():
    train_file = open('train.txt', 'r')
    samples = train_file.readlines()
    train_file.close()
    img, label = parse_line(samples[82])
    h, w, _ = img.shape
    print(label)
    cv2.circle(img, tuple([int(w * label[0]), int(h * label[1])]), 2, (0, 0, 255), 3)
    cv2.imshow('hehe', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    _run()
    # check_data()
