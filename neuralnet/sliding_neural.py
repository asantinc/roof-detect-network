    def sliding_convolution(self):
        '''
        Classify patches of an image using neural network. If viola is True, the patches are fed from ViolaJones 
        algorithm. Otherwise, the entire image is processed
        '''
        print '********* PREDICTION STARTED *********\n'
        bound_rects = list()
        if self.viola_process:
            #for each contour, do a sliding window detection
            contours = self.all_contours[self.img_name]
            for cont in contours:
                bound_rects.append(cv2.boundingRect(cont))
        else:
            c, rows, cols = self.image.shape
            bound_rects.append((0,0,cols,rows))

        for x,y,w,h in bound_rects:
            vert_patches = ((h - utils.PATCH_H) // self.step_size) + 1  #vert_patches = h/utils.PATCH_H

            #get patches along roof height
            for vertical in range(vert_patches):
                y_pos = y+(vertical*self.step_size)
                #along roof's width
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)

            #get patches from the last row also
            if (h % utils.PATCH_H>0) and (h > utils.PATCH_H):
                leftover = h-(vert_patches*utils.PATCH_H)
                y_pos = y_pos-leftover
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)



    def classify_horizontal_patches(self, patch=None, y_pos=-1):
        '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
        '''
        #roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

        x,y,w,h = patch
        hor_patches = ((w - utils.PATCH_W) // self.step_size) + 1  #hor_patches = w/utils.PATCH_W

        for horizontal in range(hor_patches):
            
            #get cropped patch
            x_pos = x+(horizontal*self.step_size)
            full_patch = self.image[:, y_pos:y_pos+utils.PATCH_H, x_pos:x_pos+utils.PATCH_W]
            full_patch = self.experiment.scaler.transform2(full_patch)

            diff = (utils.PATCH_W-utils.CROP_SIZE)/2
            candidate = full_patch[:, diff:diff+utils.CROP_SIZE, diff:diff+utils.CROP_SIZE]
           
            if candidate.shape != (3,32,32):
                print 'ERROR: patch too small, cannot do detection\n'
                continue
            #do network detection, add additional singleton dimension
            prediction = self.experiment.net.predict(candidate[None, :,:,:])
            if prediction[0] != utils.NON_ROOF:
                if prediction[0] == utils.METAL:
                    color = (255,255,255)
                elif prediction[0] == utils.THATCH:
                    color = (0,0,255)
                cv2.rectangle(self.image_detections, (x_pos+4, y_pos+4), (x_pos+utils.PATCH_H-4, y_pos+utils.PATCH_H-4), color, 1)



