
            '''
            with Timer() as t:
                #set detections and score
                for roof_type in utils.ROOF_TYPES:
                    if self.groupThreshold > 0 and roof_type == 'metal':
                        #need to covert to rectangles
                        boxes = utils.get_bounding_boxes(np.array(classified_detections[roof_type]))
                        grouped_boxes, weights = cv2.groupRectangles(np.array(boxes).tolist(), self.groupThreshold)
                        classified_detections[roof_type] = utils.convert_detections_to_polygons(grouped_boxes) 
                        #convert back to polygons

                    elif self.groupBounds and roof_type == 'metal':
                        #grouping with the minimal bound of all overlapping rects
                        classified_detections[roof_type] = self.group_min_bound(classified_detections[roof_type], img_shape[:2], erosion=self.erosion)

                    elif self.suppress is not None  and roof_type == 'metal':
                        #proper non max suppression from Felzenszwalb et al.
                        classified_detections[roof_type] = self.non_max_suppression_rects(classified_detections[roof_type], probs_of_roofs_only[roof_type])
            
            print 'Grouping took {} seconds'.format(t.secs)
            neural_time += t.secs 
            #self.print_detections(classified_detections, img_name, '_nonMax')
            '''

