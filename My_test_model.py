#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

TOTAL_IMAGE_START = 1;
TOTAL_IMAGE_NUM = 43;
# Image_source_dir = "/auto/extra/b02902015/Drone_Video/Front_Side/";
# Stitch_source_dir = "/auto/extra/b02902015/panorama-stitching/Stitch_Front/";
# Detect_output_dir = '/auto/extra/b02902015/py-faster-rcnn/Car_front_output/';
# Result_dir = '/auto/extra/b02902015/Result/Front/';

# Image_source_dir = "/auto/extra/b02902015/Drone_Video/Side/";
# Stitch_source_dir = "/auto/extra/b02902015/panorama-stitching/Stitch_Side/";
# Detect_output_dir = '/auto/extra/b02902015/py-faster-rcnn/Side_output/';
# Result_dir = '/auto/extra/b02902015/Result/Side/';

Image_source_dir = "/auto/extra/b02902015/Drone_Video/AfterCut_Rewine/";
Stitch_source_dir = "/auto/extra/b02902015/panorama-stitching/Stitch_output_Cut_Rewine/";
Detect_output_dir = '/auto/extra/b02902015/py-faster-rcnn/Side_close_output/';
Result_dir = '/auto/extra/b02902015/Result/Side_Close/';
Handle_START = 1;


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def vis_detections(im, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    # print im.shape;
    # im = im[:, 1024:2047, :]
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    print type(ax);
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # remove box too small
        print bbox;

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(Detect_output_dir + image_name);

def demo(net, image_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(image_dir, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if(cls == 'car'):
            # vis_detections(im, image_name, cls, dets, thresh=CONF_THRESH)
            return dets;

def My_vis (im, im_name, img_dets, handle_set, Overlap_Set, Handle_Overlap_Set, thresh, Car_num):
    """Draw detected bounding boxes."""
    if len(img_dets) == 0:
        return
    
    class_name = 'car';
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    car_num = 0;
    for i in handle_set:
        dets = img_dets[im_name];
        inds = np.where(dets[:, -1] >= thresh)[0];
        if(len(inds) == 0):
            continue;
        # ===========Should be modify!============
        # if(i == 5): 
        #     overlap_len = 0;
        # else:
        #     overlap_len = pre_overlap_bbox;
        # ========================================
        # ===========New idea for modify==============
        im_pre = cv2.imread(Image_source_dir + Handle_Name(i-1));
        if(i == Handle_START):
            Handle_Overlap_Set[im_name] = Handle_Overlap_Set[im_name];
        elif(Car_num[i-1] == 0):
            Handle_Overlap_Set[im_name] -= max(0, im_pre.shape[1] - Handle_Overlap_Set[Handle_Name(i-1)] + 30);
        
        print "Overlap between %d and %d: %d" % (i-1, i, Handle_Overlap_Set[im_name]);
        overlap_len = Handle_Overlap_Set[im_name];
        Jump_len = im_pre.shape[1] - Overlap_Set[im_name];
        # ============================================

        for j in inds:
            bbox = dets[j, :4]
            score = dets[j, -1]
            # remove box overlap
            Car_len_ratio = overlap_len - bbox[0] / bbox[2] - bbox[0];
            if(bbox[0] > overlap_len or i == TOTAL_IMAGE_START):
                car_num += 1;
            elif(bbox[2] > overlap_len and bbox[2] - overlap_len > overlap_len - bbox[0]):
                car_num += 1;
            print bbox;
            ax.add_patch(
                plt.Rectangle((bbox[0] + Jump_len, bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0] + Jump_len, bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw();
    plt.savefig(Result_dir + im_name);
    # plt.show();
    return car_num;

def Handle_Name(i):
    if(i < 10):
        im_name = '000' + str(i) + '.jpg';
    else:
        im_name = '00' + str(i) + '.jpg';
    return im_name;


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    print cfg.DATA_DIR;
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # My testing=========
    Handle_Overlap_Set = {};
    Overlap_Set = {};
    Stitch_info = open(Stitch_source_dir + "Stitch_info.txt", "r");
    for line in Stitch_info:
        line = line.strip().split(" ");
        Overlap_Set[line[0]] = int(line[1]);
        Handle_Overlap_Set[line[0]] = int(line[1]);
        # print line;
    img_dets = {};
    Car_num = [0] * (TOTAL_IMAGE_NUM+1);
    handle_set = [];
    f = open(Result_dir + 'Result.txt', 'w');
    # ===================
    for i in range(TOTAL_IMAGE_START, TOTAL_IMAGE_NUM + 1):
        im_name = Handle_Name(i);

        # im_name = 'pano.jpg';
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for ' + Image_source_dir + im_name;
        img_dets[im_name] = demo(net, Image_source_dir, im_name);
        if(not Handle_Overlap_Set.has_key(im_name)):
            continue;
        if(not Handle_Name(i - 1) in Handle_Overlap_Set):
            Handle_Overlap_Set[Handle_Name(i)] = 0;
            Handle_START = i;
        blend_img = cv2.imread(Stitch_source_dir + im_name);
        handle_set = [i];
        Car_num[i] += My_vis (blend_img, im_name, img_dets, handle_set, Overlap_Set, Handle_Overlap_Set, 0.8, Car_num);
        print "Car_num between %d and %d: %d" % (i-1, i, Car_num[i]);
        f.write("Car_num between %d and %d: %d\n" % (i-1, i, Car_num[i]));

    # plt.show()
