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

###################################
#VIOLA TRAINING
###################################
#Viola constants
RECTIFIED_COORDINATES = '../data/source/bounding_inhabited/'
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
                    path.append('(neural/')
                elif viola:
                    path.append('viola/')
                else:
                    path.append('source/')
            elif data_fold==VALIDATION:
                path.append('validation/')
            elif data_fold==TESTING:
                path.append('testing/')
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


def get_img_size(path):
    for i, f_name in enumerate(os.listdir(path)):
        img = cv2.imread(path+f_name)
        print img.shape[0], img.shape[1]
        
        if i%20==0:
            pdb.set_trace()


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


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


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


def rotate_point_no_translate(pos, img_rot, theta, dst_img_rows=1200, dst_img_cols=2000):
    '''
    Pos: point to be rotated, in (x,y) order
    img_rot: image around whose center the point should be rotated
    dst_img_rows, dst_img_cols: the shape of the image to which we are sendind the point
    '''
    theta = math.radians(theta)

    #translation: how much the image has moved
    oy, ox = tuple(np.array(img_rot.shape[:2])/2)
    px, py = pos

    #rotate the point
    p_x = math.cos(theta) * (px-ox) - math.sin(theta) * (py-oy) + ox
    p_y = math.sin(theta) * (px-ox) + math.cos(theta) * (py-oy) + oy
    return int(p_y), int(p_x)


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
	return rect


def four_point_transform(image, pts, rectify=False):
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

def convert_rect_to_polygon(rect):
    x, y, w, h = rect
    p1, p2, p3, p4 = (x, y), (x, y+h), (x+w, y), (x+w, y+h)
    polygon = order_points(np.array([p1, p2, p3, p4]))
    return polygon


def rotate_polygon(polygon, img, angle):
    '''
    Rotate each of the points of a polygon
    img: the image center around which we will be rotating
    '''
    rot_polygon = np.empty((polygon.shape))
    for i, pnt in enumerate(polygon):
        rot_polygon[i, :] = rotate_point(pnt, img, angle)
        print pnt
        print rot_polygon[i, :]
    #pdb.set_trace()
    #reordered_rot_polygon = order_points(rot_polygon) 
    reordered_rot_polygon=rot_polygon
    return reordered_rot_polygon


def draw_polygon(polygon, img):
    w, h = img.shape[:2]
    #polygon = order_points(polygon)
    if polygon.shape[0] == 4:
        for pnt in polygon:
            x, y = pnt
            if (x<0) or (x>w) or (y<0) or (y>h):
                #pdb.set_trace()
                print 'Polygon falls off the image'
                print polygon
                #return
        #pdb.set_trace()
        polygon = np.array(polygon, dtype='int32')
        cv2.polylines(img, [polygon], 1, (0,0,255), 5)
        #pdb.set_trace()
        #cv2.fillConvexPoly(img, polygon, 2)
    else:
        raise ValueError('draw_polygon was given a non-square polygon')



if __name__ == '__main__':
    img = cv2.imread('../data/inhabited/0001.jpg')

    # angle = 90
    # rimg = rotate_image_RGB(img, angle)
    # pnt_1 = (700,900)
    # cv2.line(rimg, pnt_1, (pnt_1[0]+1,pnt_1[1]+1), (0,255,0), 10) 
    # cv2.imwrite('../../delete/rotated_1.jpg', rimg)

    # img_normal = cv2.imread('../data/inhabited/0001.jpg')
    # px_1, py_1 = rotate_point(pnt_1, rimg, angle)
    # cv2.line(img_normal, (px_1,py_1), (px_1+1,py_1+1), (0,0,255), 10) 
    # cv2.imwrite('../../delete/rotated_2.jpg', img_normal)


    # RECTANGLE
    for angle in [45, 90, 135]:
    #for angle in  [45]:
        for posx in [100, 200, 500, 700, 1000, 1300, 1800]:
        #for posx in [500]:
            for posy in [100, 200, 500, 700, 1000, 1200]:
            #for posy in [1200]:
                rimg = rotate_image_RGB(img, angle)
                rect_1 = (posx, posy, 50, 70)
                #get polygon
                polygon_1 = convert_rect_to_polygon(rect_1)
                polygon_1 = np.array(polygon_1, dtype='int32')
                cv2.polylines(rimg, [polygon_1], 1, (0,255,0), 5)
                cv2.imwrite('../../delete/A_rotated_{0}_{1}_{2}_3.jpg'.format(angle, posx, posy), rimg)

                img_normal = cv2.imread('../data/inhabited/0001.jpg')
                polygon_2 = rotate_polygon(polygon_1, rimg, angle)

                draw_polygon(polygon_2, img_normal)
                cv2.imwrite('../../delete/A_rotated_{0}_{1}_{2}_4.jpg'.format(angle, posx, posy), img_normal)


      #TEST: draw a rect with 

    # polygon_1 = rotate_polygon(polygon, rimg, angle)
    # polygon_1 = np.array(polygon_1, dtype='int32')
    # print polygon_1
    # cv2.polylines(img_normal, [polygon_1], 1, (0,0,0), 5)

    # for point in polygon:
    #   #rotate it
    #   rot_point = rotate_point(point, rimg, angle)
    #   cv2.line(img_normal, rot_point, (rot_point[0]+1,rot_point[1]+1), (0,0,255), 10) 
    # cv2.imwrite('../../delete/rotated_test_FIXED.jpg', img_normal)



