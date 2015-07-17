import cv2
import pdb
import experiment_settings as settings



if __name__ == '__main__':
    for i in range(10):
        img_path = settings.INHABITED_PATH+'000'+str(i+1)+'.jpg'
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        assert img is not None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('my_gray.jpg', gray)

        cascade_name = 'cascade_metal_0_tall_augm0_num59_w25_h50_GITHUB' 
        #cascade_name  = 'cascade_metal_0_square_augment_num3088_w24_h24_OLD'
        detector = cv2.CascadeClassifier('../viola_jones/'+cascade_name+'/cascade.xml')    
        print detector
        detected_roofs = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        print 'A', str(len(detected_roofs))

        
        cascade_name  = 'cascade_metal_5_wide_augment_num2184_w24_h12_OLD'
        detector = cv2.CascadeClassifier('../viola_jones/'+cascade_name+'/cascade.xml')    
        print detector
        detected_roofs = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        print 'B', str(len(detected_roofs))
