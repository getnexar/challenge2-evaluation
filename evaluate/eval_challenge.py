# --------------------------------------------------------------------------------------------------
#
#   ITAY : Need to change header
#   Copyright (c) 2016-2017. Nexar Inc. - All Rights Reserved. Proprietary and confidential.
#
#   Unauthorized copying of this file, via any medium is strictly prohibited.
#
# --------------------------------------------------------------------------------------------------
from __future__ import print_function

import os
import csv
import argparse

DEBUG = False

class Box:
    def __init__(self,csv_row = None, img_name=None):
        self.x0 = csv_row[1]
        self.y0 = csv_row[2]
        self.x1 = csv_row[3]
        self.y1 = csv_row[4]
        self.label = csv_row[5]
        self.confidence = csv_row[6]
        self.img_name = img_name
        self.is_matched = False

    def __str__(self):
        out_str = '[{},{}] x [{},{}] : {} (confidence={})'.format(self.x0,self.x1,self.y0,self.y1,self.label,self.confidence)
        return out_str

    def __lt__(self, other):
         return self.confidence < other.confidence

class EvalDetector:
    def __init__(self, conf, precision, recall):
        self._conf = conf
        self._precision = precision
        self._recall = recall

    def AP(self):
        """
        Compute average precision for submission to Nexar's second challenge
        :return: Average precision (float)
        """
        prev_recall_value = 0.0
        ap = 0.0
        for conf, precision, recall in zip(self._conf,self._precision,self._recall):
            delta = recall - prev_recall_value
            ap += precision * delta
            prev_recall_value = recall
        return ap



def read_boxes_from_csv(csv_file):
    boxes = {}
    img = []
    with open(csv_file, 'rt') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i > 0:
                img_name = row[0]
                img.append(img_name)
                row[1:5] = [float(e) for e in row[1:5]]
                row[6] = float(row[6])
                if img_name in boxes:
                    boxes[img_name].append(Box(csv_row=row,img_name=img_name))
                else:
                    boxes[img_name] = [Box(csv_row=row,img_name=img_name)]

    return boxes


def iou(box1, box2):
    lr = (min(box1.x1,box2.x1) - max(box1.x0, box2.x0)) + 1
    if lr > 0:
        tb = (min(box1.y1,box2.y1) - max(box1.y0, box2.y0)) + 1

        if tb > 0:
            intersection = tb * lr
            w1 = box1.x1 - box1.x0 + 1
            h1 = box1.y1 - box1.y0 + 1
            w2 = box2.x1 - box2.x0 + 1
            h2 = box2.y1 - box2.y0 + 1
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection) / float(union)

    return 0.0


def find_best_iou_box(detector_box, ground_truth_boxes):
    best_iou = -1
    best_i = -1
    for i, gt_box in enumerate(ground_truth_boxes):
        if not gt_box.is_matched:
            computed_iou = iou(detector_box,gt_box)
            if  computed_iou > best_iou:
                best_iou = computed_iou
                best_i = i

    return best_i, best_iou

def eval_detector_csv(gt_csv_file, detector_csv_file, iou_threshold):
    ground_truth_boxes_by_img = read_boxes_from_csv(gt_csv_file)
    if DEBUG:
        print ('Ground truth: cvs - {} , Number images = {}'.format(gt_csv_file,len(ground_truth_boxes_by_img)))

    detector_boxes_by_img = read_boxes_from_csv(detector_csv_file)
    if DEBUG:
        print ('Detector: cvs - {} , Number images= {}'.format(detector_csv_file,len(detector_boxes_by_img)))

    n_empty_gt_images = 0
    for img_obj in detector_boxes_by_img:
        if img_obj not in ground_truth_boxes_by_img:
            if DEBUG:
                print ('ERROR: detected image name not a ground truth image: {}'.format(img_obj))
            n_empty_gt_images += 1
            continue
    if DEBUG:
        print('Number of empty GT images: {}'.format(n_empty_gt_images))

    return eval_detector(ground_truth_boxes_by_img, detector_boxes_by_img, iou_threshold)


def eval_detector(ground_truth_boxes_by_img, detector_boxes_by_img, iou_threshold):
    n_ground_truth_boxes = 0
    for img in ground_truth_boxes_by_img:
        n_ground_truth_boxes += len(ground_truth_boxes_by_img[img])

    all_detected_boxes = []
    for d in detector_boxes_by_img.values():
        all_detected_boxes += d
    all_detected_boxes.sort(reverse=True)

    n_correct_detected_boxes = 0
    n_detected_boxes = 0
    conf = [0.0] * len(all_detected_boxes)
    precision = [0.0] * len(all_detected_boxes)
    recall = [0.0] * len(all_detected_boxes)
    for k, detector_box in enumerate(all_detected_boxes):
        n_detected_boxes += 1
        if detector_box.img_name in ground_truth_boxes_by_img:
            ground_truth_boxes = ground_truth_boxes_by_img[detector_box.img_name]
        else:
            ground_truth_boxes = []

        best_ground_truth_i, best_ground_truth_iou = find_best_iou_box(detector_box, ground_truth_boxes)
        is_match = (best_ground_truth_i >=0) and  (best_ground_truth_iou >= iou_threshold)
        if is_match:
            # detector_box matched to ground truth box
            ground_truth_boxes[best_ground_truth_i].is_matched = True
            n_correct_detected_boxes += 1

        conf[k] = detector_box.confidence
        precision[k] = float(n_correct_detected_boxes)/float(n_detected_boxes)
        recall[k] = float(n_correct_detected_boxes)/float(n_ground_truth_boxes)

    edetect = EvalDetector(conf,precision,recall)

    return edetect.AP()




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', help='Ground truth csv file', action='store')
    parser.add_argument('-d', help='Detector csv file', action='store')

    args = parser.parse_args()

    gt_csv =  args.g
    dt_csv = args.d


    if not os.path.isfile(gt_csv):
        print ('Can not find ground truth csv file: {}'.format(gt_csv))
        exit(-1)

    if not os.path.isfile(dt_csv):
        print ('Can not find ground truth csv file: {}'.format(dt_csv))
        exit(-1)

    iou_threshold = 0.75
    print ('{}'.format(eval_detector_csv(gt_csv, dt_csv, iou_threshold)))




if __name__ == "__main__":
    main()

