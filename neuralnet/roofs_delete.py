import cv2
from get_data import DataLoader, Roof
import os
import pdb
import extract_rect
import numpy as np

out_path = '../../../marked_roofs/'
for img_name in os.listdir('../data/inhabited/'):
	if img_name.endswith('.jpg'):
		img = cv2.imread('../data/inhabited/'+img_name)	
		assert img is not None
		roofs = DataLoader().get_roofs_new_dataset('../bounding_inhabited/'+img_name[:-3]+'xml')
		for roof in roofs:
			if type(roof) is Roof:
				cv2.rectangle(img, (roof.xmin, roof.ymin), (roof.xmin+roof.width, roof.ymin+roof.height), (0,255,0), 2)
			else:
				roof = np.array(roof)
				pts = extract_rect.order_points(roof)
				cv2.rectangle(img, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0,255,0), 2)
		cv2.imwrite(out_path+img_name, img)