# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import pdb
import cv2
import datetime

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    #obj_struct['pose'] = obj.find('pose').text
    #obj_struct['truncated'] = int(obj.find('truncated').text)
    #obj_struct['difficult'] = int(obj.find('difficult').text)

    obj_struct['pose'] = 'Unspecified' if obj.find('pose') == None else obj.find('pose').text
    obj_struct['truncated'] = 0 if obj.find('truncated') == None else int(obj.find('truncated').text)
    obj_struct['difficult'] = 0 if obj.find('difficult') == None else int(obj.find('difficult').text)

    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def adas_ap(rec, prec, use_07_metric=False):
  """ ap = adas_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def adas_eval(detpath,
             imagepath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = adas_eval(detpath,
                              imagepath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]
  
  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'w') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    difficult = np.array([x.get('difficult', 0) for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}
  
  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    print('read detfile {}'.format(detfile))
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  iou = np.zeros(nd)
  l1 = np.zeros((nd,4))
  area = np.zeros(nd)

  fpbbs = {}
  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)
      area[d] = (bb[2] - bb[0] + 1)*(bb[3] - bb[1] + 1)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      iou[d] = 0.0 if ovmax == -np.inf else ovmax
      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
            l1[d] = np.absolute(BBGT[jmax] - bb)
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.
      
      if fp[d]:
          if fpbbs.get(image_ids[d]) is None:
            fpbbs[image_ids[d]] = [bb] 
          else:
            fpbbs[image_ids[d]].append(bb)

    vis_fp(imagepath, fpbbs)

  # compute precision recall
  fp_c = np.cumsum(fp)
  tp_c = np.cumsum(tp)
  l1_c = np.cumsum(l1, axis=0)
  rec = tp_c / float(npos)

  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp_c / np.maximum(tp_c + fp_c, np.finfo(np.float64).eps)
  ap = adas_ap(rec, prec, use_07_metric)

  now = datetime.datetime.now().strftime('%m%d%H%M')
  with open('eval_{}.txt'.format(now),'w') as out:
    for i in range(nd):
      tp_l1_mean = l1_c[i]/tp_c[i]
      out.write("gt: %d tp: %d fp: %d iou: %.3f area: %-5d tp_c: %-4d fp_c: %-4d score: %.3f pre: %.3f rec: %.3f, l1_x0: %.3f, l1_y0: %.3f, l1_x1: %.3f, l1_y2:%.3f, image: %-20s\n" \
          % (npos, tp[i], fp[i], iou[i], int(area[i]), tp_c[i], fp_c[i], -sorted_scores[i], prec[i], rec[i], tp_l1_mean[0], tp_l1_mean[1], tp_l1_mean[2], tp_l1_mean[3], os.path.basename(image_ids[i])))
      '''
      out.write("gt: %d tp: %d fp: %d iou: %.3f tp_c: %-4d fp_c: %-4d score: %.3f pre: %.3f rec: %.3f, l1_x0: %.3f, l1_y0: %.3f, l1_x1: %.3f, l1_y2:%.3f\n" \
          % (npos, tp[i], fp[i], iou[i], tp_c[i], fp_c[i], -sorted_scores[i], prec[i], rec[i], tp_l1_mean[0], tp_l1_mean[1], tp_l1_mean[2], tp_l1_mean[3]))
      '''

  return rec, prec, ap

def vis_fp(imagepath, fpbbs):
    if not os.path.exists('fps'):
        os.mkdir('fps')

    for image_id in fpbbs:
        image_path = imagepath.format(image_id)
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        for bb in fpbbs[image_id]:
            x_lt,y_lt,x_rb,y_rb = [int(e) for e in bb[0:4]]
            cv2.rectangle(img, (x_lt,y_lt), (x_rb,y_rb), (0,255,0), 1)
        cv2.imwrite(os.path.join('fps', image_name), img)
