import cv2
from get_data import DataLoader
import os
import pdb


out_path = '../../../marked_roofs/'
for img_name in os.listdir('../data/inhabited/'):
	if img_name.endswith('.jpg'):
		img = cv2.imread('../data/inhabited/'+img_name)	
		roofs = DataLoader().get_roofs('../data/inhabited/'+img_name[:-3]+'xml', img_name)
		for roof in roofs:
			cv2.rectangle(img, (roof.xmin, roof.ymin), (roof.xmin+roof.width, roof.ymin+roof.height), (0,255,0), 2)
		cv2.imwrite(out_path+img_name, img)