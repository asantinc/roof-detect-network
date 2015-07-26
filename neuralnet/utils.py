from collections import OrderedDict
import os
import subprocess
from datetime import datetime
import sys
import pdb
import datetime
import csv
import math
import cv2
import numpy as np
import warnings

###################################
#types of roof
###################################
NON_ROOF = 0
METAL = 1
THATCH = 2
ROOF_LABEL = {'background':NON_ROOF, 'metal':METAL, 'thatch':THATCH}

#Constants for image size
IMG_SIZE = 40
CROP_SIZE = 32
PATCH_W = PATCH_H = 40

TRAINING = 1
VALIDATION = 2
TESTING = 3
SMALL_TEST = 4

IN = 1  #input 
OUT = 2 #output

VIOLA_ANGLES = [0, 45, 90, 135]
VOC_threshold = 0.5 

ROOF_TYPES = ['metal', 'thatch']

###################################
#VIOLA TRAINING
###################################
#Viola constants
RECTIFIED_COORDINATES = '../data/source/bounding_inhabited/'
UNINHABITED_PATH = '../data/source/uninhabited/'
BG_FILE = '../viola_jones/bg.txt'
DAT_PATH = '../viola_jones/all_dat/'
VEC_PATH = '../viola_jones/vec_files/'
CASCADE_PATH = '../viola_jones/cascades/'


###################################
#NEURAL TRAINING PATCHES
###################################
TRAINING_NEURAL_POS = '../data/training/neural/positives/'  #where the true positives from training source are 
TRAINING_NEURAL_NEG = '../data/training/neural/negatives/'  

#Constants for training neural network
#NET_PARAMS_PATH = "../parameters/net_params/"
OUT_REPORT = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/report_'  
OUT_HISTORY = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/history_' 
OUT_IMAGES = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/images_'

FTRAIN = '../data/training/'
FTRAIN_LABEL = '../data/training/labels.csv'
#TEST_PATH = '../data/test/'


###################################
#Constants for debugging
###################################
VERBOSITY = 1   #varies from 1 to 3
DEBUG = False

###################################
#Pipeline
###################################
STEP_SIZE = PATCH_H #used by detection pipeline


########################################
# PATH CONSTRUCTION
########################################

def time_stamped(fname=''):
    if fname != '':
        fmt = '{fname}_%m-%d-%H-%M/'
    else:
        fmt =  '%m-%d-%H-%M/'
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def mkdir(out_folder_path=None, confirm=False):
    assert out_folder_path is not None
    if not os.path.isdir(out_folder_path):
        if confirm:
            answer = raw_input('{0} does not exist. Do you want to create it? Type "y" or "n"'.format(out_folder_path))
            if answer == 'n':
                sys.exit(-1)
        subprocess.check_call('mkdir {0}'.format(out_folder_path), shell=True)
    else:
        overwrite = raw_input('Folder {0} exists; overwrite, y or n?'.format(out_folder_path))
        if overwrite == 'n':
            sys.exit(-1)
    print 'The following folder has been created: {0}'.format(out_folder_path)


def check_append(path_being_constructed, addition):
    path_being_constructed.append(addition)

    if os.path.isdir(''.join(path_being_constructed)):
        return path_being_constructed
    else:
        mkdir(out_folder_path=''.join(path_being_constructed), confirm=True) 


def get_path(in_or_out=None, out_folder_name=None, params=False, full_dataset=False, 
                    pipe=False, viola=False, template=False, neural=False, data_fold=None):
    '''
    Return path to either input, output or parameter files. If a folder does not exit, ask user for confirmation and create it
    '''
    path = list()
    if params:
        #PARAMETER FILES
        check_append(path, '../parameters/')
        if neural:
            check_append(path, 'net_params/') 
        elif pipe:
            check_append(path, 'pipe_params/')
        elif viola:
            check_append(path, 'detector_combos/')
        else:
            raise ValueError('There are no parameter files for templating')
    else:
        assert in_or_out is not None
        if in_or_out == IN:
            #INPUT PATH
            if full_dataset==False:
                path.append('../data_original/')
            else:
                path.append('../data/')
            assert data_fold is not None
            if data_fold == TRAINING:
                path.append('training/')
                if neural:
                    path.append('neural/')
                elif viola:
                    path.append('viola/')
                else:
                    path.append('source/')
            elif data_fold==VALIDATION:
                path.append('validation/')
            elif data_fold==TESTING:
                path.append('testing/')
            elif data_fold==SMALL_TEST:
                path.append('small_test/')
            else:
                raise ValueError
        elif in_or_out == OUT:
            #OUTPUT FOLDERS
            check_append(path,'/afs/inf.ed.ac.uk/group/ANC/s0839470/output/')
            if viola:
                check_append(path,'viola/')
            elif template:
                check_append(path,'templating/')
            elif neural:
                check_append(path,'neural/')
            elif pipe:
                check_append(path,'pipe/')
            else:
                raise ValueError('Cannot create path. Try setting viola, pipe, template or neural to True to get a path')
            
            original = 'original_' if full_dataset == False else ''
            if data_fold==TRAINING:
                check_append(path,'with_{0}training_set/'.format(original))
            elif data_fold==TESTING:
                check_append(path,'with_{0}testing_set/'.format(original))
            elif data_fold==VALIDATION:
                check_append(path,'with_{0}validation_set/'.format(original))
            elif data_fold==SMALL_TEST:
                check_append(path,'with_small_test/')
            else:
                raise ValueError('Even though you requested the path of an output folder, you have to specify if you created the output from training, validation or testing data')
        else:
            raise ValueError('Must specify input or output folder.')
        if out_folder_name is not None:
            if out_folder_name.endswith('/'):
                check_append(path, out_folder_name)
            else:
                check_append(path, out_folder_name+'/')
    return ''.join(path)
     

def print_debug(to_print, verbosity=1):
    #Print depending on verbosity level
    if verbosity <= VERBOSITY:
        print str(to_print)

########################################
# Getting parameters 
########################################

def get_params_from_file(path):
    print 'Getting parameters from {0}'.format(path)
    parameters = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for par in reader:
            if len(par) == 0:
                break
            elif len(par)== 2:
                parameters[par[0].strip()] = par[1].strip()

            elif len(par) == 3:
                if par[0].strip() not in parameters:
                    parameters[par[0].strip()] = dict()
                parameters[par[0].strip()][par[1].strip()] = par[2].strip() 
            else:
                raise ValueError('Cannot process {0}'.format(path))
    return parameters

########################
# Image Resize 
########################

def resize_rgb(img, w=PATCH_W, h=PATCH_H):
    resized = np.empty((h,w,3))
    for channel in range(img.shape[-1]):
        resized[:,:,channel] = cv2.resize(img[:,:,channel], (h, w), dst=resized[0,:,:], fx=0, fy=0, interpolation=cv2.INTER_AREA) 
    return resized


########################
# Image rotation
########################

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M


def sum_mask(array):
    '''Sum all ones in a binary array
    '''
    return np.sum(np.sum(array, axis=0), axis=0)


def rotate_image(image, angle):
  '''Rotate image "angle" degrees.

  How it works:
    - Creates a blank image that fits any rotation of the image. To achieve
      this, set the height and width to be the image's diagonal.
    - Copy the original image to the center of this blank image
    - Rotate using warpAffine, using the newly created image's center
      (the enlarged blank image center)
    - Translate the four corners of the source image in the enlarged image
      using homogenous multiplication of the rotation matrix.
    - Crop the image according to these transformed corners
  '''

  diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
  offset_x = (diagonal - image.shape[0])/2
  offset_y = (diagonal - image.shape[1])/2
  #dst_image = np.zeros((diagonal, diagonal, 3), dtype='uint8')
  dst_image = np.zeros((diagonal, diagonal), dtype='uint8')
  image_center = (diagonal/2, diagonal/2)

  R = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  #dst_image[offset_x:(offset_x + image.shape[0]), \
  #          offset_y:(offset_y + image.shape[1]), \
  #          :] = image
  dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1])] = image
  dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)
  # Calculate the rotated bounding rect
  x0 = offset_x
  x1 = offset_x + image.shape[0]
  x2 = offset_x
  x3 = offset_x + image.shape[0]

  y0 = offset_y
  y1 = offset_y
  y2 = offset_y + image.shape[1]
  y3 = offset_y + image.shape[1]

  corners = np.zeros((3,4))
  corners[0,0] = x0
  corners[0,1] = x1
  corners[0,2] = x2
  corners[0,3] = x3
  corners[1,0] = y0
  corners[1,1] = y1
  corners[1,2] = y2
  corners[1,3] = y3
  corners[2:] = 1

  c = np.dot(R, corners)

  x = int(c[0,0])
  y = int(c[1,0])
  left = x
  right = x
  up = y
  down = y

  for i in range(4):
    x = int(c[0,i])
    y = int(c[1,i])
    if (x < left): left = x
    if (x > right): right = x
    if (y < up): up = y
    if (y > down): down = y
  h = down - up
  w = right - left

  #cropped = np.zeros((w, h, 3), dtype='uint8')
  cropped = np.zeros((w, h), dtype='uint8')
  #cropped[:, :, :] = dst_image[left:(left+w), up:(up+h), :]
  cropped[:, :] = dst_image[left:(left+w), up:(up+h)]
  return cropped

def rotate_image_RGB(image, angle):
  '''Rotate image "angle" degrees.

  How it works:
    - Creates a blank image that fits any rotation of the image. To achieve
      this, set the height and width to be the image's diagonal.
    - Copy the original image to the center of this blank image
    - Rotate using warpAffine, using the newly created image's center
      (the enlarged blank image center)
    - Translate the four corners of the source image in the enlarged image
      using homogenous multiplication of the rotation matrix.
    - Crop the image according to these transformed corners
  '''

  diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
  offset_x = (diagonal - image.shape[0])/2
  offset_y = (diagonal - image.shape[1])/2
  dst_image = np.zeros((diagonal, diagonal, 3), dtype='uint8')
  #dst_image = np.zeros((diagonal, diagonal), dtype='uint8')
  image_center = (diagonal/2, diagonal/2)

  R = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  dst_image[offset_x:(offset_x + image.shape[0]), \
            offset_y:(offset_y + image.shape[1]), \
            :] = image
  #dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1])] = image
  dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)
  # Calculate the rotated bounding rect
  x0 = offset_x
  x1 = offset_x + image.shape[0]
  x2 = offset_x
  x3 = offset_x + image.shape[0]

  y0 = offset_y
  y1 = offset_y
  y2 = offset_y + image.shape[1]
  y3 = offset_y + image.shape[1]

  corners = np.zeros((3,4))
  corners[0,0] = x0
  corners[0,1] = x1
  corners[0,2] = x2
  corners[0,3] = x3
  corners[1,0] = y0
  corners[1,1] = y1
  corners[1,2] = y2
  corners[1,3] = y3
  corners[2:] = 1

  c = np.dot(R, corners)

  x = int(c[0,0])
  y = int(c[1,0])
  left = x
  right = x
  up = y
  down = y

  for i in range(4):
    x = int(c[0,i])
    y = int(c[1,i])
    if (x < left): left = x
    if (x > right): right = x
    if (y < up): up = y
    if (y > down): down = y
  h = down - up
  w = right - left

  cropped = np.zeros((w, h, 3), dtype='uint8')
  #cropped = np.zeros((w, h), dtype='uint8')
  cropped[:, :, :] = dst_image[left:(left+w), up:(up+h), :]
  #cropped[:, :] = dst_image[left:(left+w), up:(up+h)]
  return cropped


#######################
# CONVERSIONS
#######################

def check_proper_rect(rect):
  ''' Checks that none of the points of a rectangle are equal to each other
  '''
  if (((rect[0][0] == rect[1][0]) and (rect[0][1] == rect[1][1])) 
          or ((rect[0][0] == rect[2][0]) and (rect[0][1] == rect[2][1]) ) 
          or ((rect[0][0] == rect[3][0]) and (rect[0][1] == rect[3][1]))
          or ((rect[1][0] == rect[2][0]) and (rect[1][1] == rect[2][1])) 
        or ((rect[1][0] == rect[3][0]) and (rect[1][1] == rect[3][1]))
        or ((rect[2][0] == rect[3][0]) and (rect[2][1] == rect[3][1])) ):
      return False
  else:
      return True

def recalculate_rect(pts):
  # If reorder pnts has failed...
  # we try to reorder based on which are the min_x and min_y coordinates
  rect = np.zeros((4, 2), dtype = "float32")

  y_coords = pts[:, 1]
  x_coords = pts[:, 0]

  rect[0] = pts[np.argmin(y_coords)]
  rect[2] = pts[np.argmax(y_coords)]
  rect[1] = pts[np.argmax(x_coords)]
  rect[3] = pts[np.argmin(x_coords)]

  if check_proper_rect(rect) == False:
    raise ValueError("The rectangle could not be ordered correctly {0}".format(rect))
  return rect


def order_points(pts):
	# reorder so that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
        if check_proper_rect(rect) == False:
            rect = recalculate_rect(pts)
	return rect


def convert_rect_to_polygon(rect):
    '''Convert rect in the form (x, y, w, h) to an array with 4 (x, y) coordinates 
    '''
    x, y, w, h = rect
    p1, p2, p3, p4 = (x, y), (x, y+h), (x+w, y), (x+w, y+h)
    polygon = order_points(np.array([p1, p2, p3, p4]))
    return polygon


def convert_detections_to_polygons(detections):
    '''Covert (x,y,w,h) to (p1, p2, p3, p4) where each point is in the order (x, y)
    '''
    polygons = np.zeros((detections.shape[0], 4, 2))
    for i, d in enumerate(detections):
        polygons[i, :] = convert_rect_to_polygon(d)
    return polygons



########################
# Perspective transform
#######################

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

#########################
# ROTATIONS
#########################

def rotate_point(pos, img_rot, theta, dst_img_rows=1200, dst_img_cols=2000):
    '''
    Pos: point to be rotated, in (x, y) order
    img_rot: image around whose center the point should be rotated
    dst_img_rows, dst_img_cols: the shape of the image to which we are sendind the point
    '''
    warnings.warn('!!!! rotate_point assumes image is size 1200,2000, but this may not always be the case. You should pass in the size of the image')
    theta = math.radians(theta)

    #translation: how much the image has moved
    oy, ox = tuple(np.array(img_rot.shape[:2])/2)
    off_y, off_x = (oy-600), (ox-1000)

    #reposition the point and the center: move it back 
    oy, ox = 600, 1000
    px, py = pos[0]-off_x, pos[1]-off_y

    #rotate the point
    p_x = math.cos(theta) * (px-ox) - math.sin(theta) * (py-oy) + ox
    p_y = math.sin(theta) * (px-ox) + math.cos(theta) * (py-oy) + oy
    return int(p_x), int(p_y)


def rotate_polygon(polygon, img, angle):
    '''Rotate each of the points of a polygon
    img: the image center around which we will be rotating
    '''
    rot_polygon = np.empty((polygon.shape))
    for i, pnt in enumerate(polygon):
        rot_polygon[i, :] = rotate_point(pnt, img, angle)
    #reordered_rot_polygon = order_points(rot_polygon)--> this messes points up
    reordered_rot_polygon = rot_polygon
    return reordered_rot_polygon


def rotate_detection_polygons(detections, img, angle, dst_img_shape, remove_off_img=False):
    '''Rotate all detections that are already in the form of polygons
    for a given angle and an original image around whose center we rotate
    '''
    rotated_polygons = np.zeros((detections.shape[0], 4, 2))
    accepted_number = 0
    h, w = dst_img_shape

    for detection in detections:
        accept_polygon=True
        rotated_pol = rotate_polygon(detection, img, angle)

        #remove points that fall off the image
        if remove_off_img == True:
            for pnt in rotated_pol:
                x, y = pnt
                if (x<0) or (y<0) or (x>w) or (y>h):
                    accept_polygon = False        
                    continue
        if accept_polygon:
            rotated_polygons[accepted_number, :] = rotate_polygon(detection, img, angle)
            accepted_number += 1
    return rotated_polygons[:accepted_number,:] 


########################
# DRAWING
########################

def draw_polygon(polygon, img, fill=False, color=(0,0,255), thickness=2, number=None):
    '''
    Draw a filled or unfilled polygon
    If number is provided, we number the roof
    '''
    w, h = img.shape[:2]
    #polygon = order_points(polygon) --> this was causing trouble
    if polygon.shape[0] == 4:
        polygon = np.array(polygon, dtype='int32')
        if fill:
            #color in this case must be an integer, not a tuple
            cv2.fillConvexPoly(img, polygon, color)
        else:
            cv2.polylines(img, [polygon], 1, color, thickness)

        if number is not None:
            top_left, _,_,_ = order_points(polygon)
            x, y = top_left
            text =  'roof {0}'.format(number)
            cv2.putText(img, text, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2) 
    else:
        raise ValueError('draw_polygon was given a non-square polygon')



def draw_detections(polygon_list, img, fill=False, color=(0, 0, 255), number=False, thickness=2):
    for i, polygon in enumerate(polygon_list):
        num = i if number else None
        draw_polygon(polygon, img, fill=fill, color=color, number=num, thickness=thickness)



if __name__ == '__main__':
    pass

