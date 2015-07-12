'''
    def append_negative_patch_arrays(self, all_roofs, all_labels, patch_no):
        '''
        #'''Append negative numpy arrays with negative examples to the all_roofs list, and append the correct non_roof label to all_labels list
'''
        print 'Getting the negative patches....\n'
        img_names = DataLoader.get_img_names_from_path(path=settings.UNINHABITED_PATH)
        negative_patches = (patch_no)/len(img_names)
        
        all_roof_objects = list()
        #Get negative patches
        for i, img_path in enumerate(img_names):
            #get random ymin, xmin, but ensure the patch will fall inside of the image
            print 'Negative image {0}'.format(i)
            try:
                img = cv2.imread(settings.UNINHABITED_PATH+img_path)
            except IOError:
                print 'Cannot open '+img_path
            else:
                h, w, _ = img.shape
                
                for p in range(negative_patches):
                    w_max = w - settings.PATCH_W
                    h_max = h - settings.PATCH_H
                    xmin = random.randint(0, w_max)
                    ymin = random.randint(0, h_max)

                    roof = Roof(xmin=xmin, ymin=ymin,xmax=xmin+settings.PATCH_W, 
                                    ymax=ymin+settings.PATCH_H, roof_type=settings.NON_ROOF)
                    roof.img_name = settings.UNINHABITED_PATH+img_path
                    try:
                        patch = img[ymin:ymin+settings.PATCH_H, xmin:xmin+settings.PATCH_W]
                    except Exception, e:
                        print e
                    else:
                        all_roof_objects.append(roof)
                        all_roofs.append(patch)
                        all_labels.append(settings.NON_ROOF)
        #return the roof arrays, the labels and the roof objects containing info about the roof patch origin 
        return all_roofs, all_labels, all_roof_objects
'''


