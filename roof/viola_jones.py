import os
import get_data
import experiment_settings as settings
import pdb

import numpy as np
import cv2

class DataAugmentation(object):
    def transform(self, Xb, yb):
        self.Xb = Xb
        self.random_flip()
        self.random_rotation(10.)
        self.random_crop()
        return self.Xb, yb

    def random_rotation(self, ang, fill_mode="nearest", cval=0.):
        angle = np.random.uniform(-ang, ang)
        self.Xb = scipy.ndimage.interpolation.rotate(self.Xb, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)

    def random_crop(self):
        #Extract 32 by 32 patches from 40 by 40 patches, rotate them randomly
        temp_Xb = np.zeros((self.Xb.shape[0],self.Xb.shape[1], CROP_SIZE, CROP_SIZE))
        margin = IMG_SIZE-CROP_SIZE
        for img in range(self.Xb.shape[0]):
            xmin = np.random.randint(0, margin)
            ymin = np.random.randint(0, margin)
            temp_Xb[img, :,:,:] = self.Xb[img, :, xmin:(xmin+CROP_SIZE), ymin:(ymin+CROP_SIZE)]
        self.Xb = temp_Xb

    def random_flip(self):
        # Flip half of the images in this batch at random:
        bs = self.Xb.shape[0]
        indices_hor = np.random.choice(bs, bs / 2, replace=False)
        indices_vert =  np.random.choice(bs, bs / 2, replace=False)
        self.Xb[indices_hor] = self.Xb[indices_hor, :, :, ::-1]
        self.Xb[indices_vert] = self.Xb[indices_vert, :, ::-1, :]


def setup_negative_samples():
    output = open('../data/bg.txt', 'w')
    for file in os.listdir(settings.UNINHABITED_PATH):
        if file.endswith('.jpg'):
            output.write(settings.UNINHABITED_PATH+file+'\n')


def get_dat_string(roof_list, img_path):
    return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'


def setup_positive_samples():
    metal_n = '../data/metal.dat'
    thatch_n = '../data/thatch.dat'

    with open(metal_n, 'w') as metal_f, open(thatch_n, 'w') as thatch_f:
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.INHABITED_PATH)
        roof_loader = get_data.DataLoader()

        for img_name in img_names_list:
            xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
            img_path = settings.INHABITED_PATH+img_name
    
            roofs, _, _ = roof_loader.get_roofs(xml_path)
            metal_log = list()
            thatch_log = list()
            for roof in roofs:
                #append roof characteristics separated by single space
                roof_info = str(roof.xmin)+' '+str(roof.ymin)+' '+str(roof.width)+' '+str(roof.height)

            	if roof.roof_type == 'metal':
                    metal_log.append(roof_info)
                elif roof.roof_type == 'thatch':
                    roof_info = str(roof.xmin)+' '+str(roof.ymin)+' '+str(30)+' '+str(30)
                    thatch_log.append(roof_info)

            if len(metal_log)>0:
                metal_f.write(get_dat_string(metal_log, img_path))
            if len(thatch_log)>0:
                thatch_f.write(get_dat_string(thatch_log, img_path))


if __name__ == '__main__':
    #setup_negative_samples()
    #setup_positive_samples()

    thatch_cascade = cv2.CascadeClassifier('../viola_jones/classifier/cascade.xml')

    img = cv2.imread('../data/inhabited/0001.jpg', flags=cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roofs = thatch_cascade.detectMultiScale(gray, 1.3, 5)
    for i, (x,y,w,h) in enumerate(roofs):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imwrite('viola.png',img)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






