import mat73
import os
import numpy as np
import cv2

def digit_class(data_path, dest_path):
    print("loading mat file.......")
    matdata = mat73.loadmat(os.path.join(data_path,"digitStruct.mat"))
    print(matdata.keys())

    for i in range(len(matdata['digitStruct']['name'])):
        try:
            file_name, boundaries = matdata['digitStruct']['name'][i], matdata['digitStruct']['bbox'][i]
            img = cv2.imread("train/"+file_name)
            try:
                for j in range(len(boundaries['height'])):
                    x1 = int(boundaries['top'][j])
                    x2 = int(boundaries['top'][j]+boundaries['height'][j])
                    y1 = int(boundaries['left'][j])
                    y2 = int(boundaries['left'][j]+boundaries['width'][j])
                    img_ = img[x1:x2,y1:y2]
                    outfile = dest_path+str(i)+"_"+str(int(boundaries['label'][j]))+".jpg"
                    cv2.imwrite(outfile,img_)
                print(i)
            except:
                pass
        except:
            pass


def non_digit_class(data_path, dest_path):
    mser = cv2.MSER_create()
    count = 0
    for img_path in os.listdir(data_path):
        n_img = cv2.imread(os.path.join(data_path,img_path))
        regions,_ = mser.detectRegions(n_img)
        for region in regions:
            y2,x2 = np.max(region,axis=0)
            y1,x1 = np.min(region, axis=0)
            nature_img = n_img[x1:x2, y1:y2]
            if nature_img.shape[0]>10 and nature_img.shape[1]>10:
                count+=1
                cv2.imwrite(dest_path+str(count)+".jpg",nature_img)


digit_class("./train/", "./data_for_detection/digits")
non_digit_class("./nature/", "./data_for_detection/non_digits")

