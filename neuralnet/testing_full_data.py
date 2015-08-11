import utils
from get_data import DataLoader
import pdb
import cv2
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    full_path = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING, full_dataset=True) 
    xml_files = [f for f in os.listdir(full_path) if f.endswith('.xml')] 
    roof_types = set()
    roof_polygons = dict()
    total_metal = 0
    total_thatch = 0
    total_tiled = 0
    for xml_file in xml_files[:10]:
        img_name = xml_file[:-4]
        roof_polygons[img_name] = DataLoader.get_all_roofs_full_dataset(merge_tiled=True, xml_name=xml_file, xml_path=full_path) 
        roof_types.update(roof_polygons[img_name].keys())

        try:
            print full_path+img_name+'.jpg'
            image = cv2.imread(full_path+img_name+'.jpg')
        except IOError as e:
            print e
            sys.exit()

        for r, roof_type in enumerate(roof_polygons[img_name].keys()):
            if roof_type == 'thatch':
                color = (255,0,0)
                total_thatch += len(roof_polygons[img_name][roof_type])
            elif roof_type == 'metal':
                color = (0,255,0)
                total_metal += len(roof_polygons[img_name][roof_type])
            elif roof_type == 'tiled':
                color = (0,0,255)
                total_tiled += len(roof_polygons[img_name][roof_type])
            print img_name
            print roof_type
            print len(roof_polygons[img_name][roof_type])

            for roof in roof_polygons[img_name][roof_type]:
                cv2.rectangle(image, (roof.xmin, roof.ymin), (roof.xmax, roof.ymax), color=color)    
            b,g,r = cv2.split(image)
            img2 = cv2.merge([r,g,b])
            plt.imshow(img2)
            plt.show()


        path = '~/delete/'
        cv2.imwrite(path+'annotated_'+img_name+'.jpg', image)

    print 'metal {}'.format(total_metal)
    print 'thatch {}'.format(total_thatch)
    print 'tiled {}'.format(total_tiled)

        


