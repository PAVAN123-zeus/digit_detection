import cv2
import operator
import numpy as np
from tqdm import tqdm

class Region_proposal_classifier():
    def __init__(self, detection_model, classification_model):
        self.detection_model = detection_model
        self.classification_model = classification_model

    def __non_max_suppression__(self, boxes, probabilities, overlap_threshold=0.3):
        if len(boxes) == 0:
            return []

        pick = []
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probabilities)
        
        while len(idxs) > 0:        
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)    
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
            
        return boxes[pick].astype("int"), probabilities[pick]
    
    def __mser__(self, gray):
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        return regions
    
    def __filter_background__(self, gray, regions):
        boxes = list()
        probabilities = list()
        
        for i, p in enumerate(regions):
            try:
                x2, y2 = np.amax(p, axis=0)
                x1, y1 = np.amin(p, axis=0)
                if (x2-x1) >= (y2-y1):
                    continue
                if (x2-x1) == 1 or (y2-y1) == 1:
                    continue
                img = gray[y1-2: y2+2, x1-2: x2+2]
                img = cv2.resize(img, (32, 32))
                img = np.reshape(img, (1, 32, 32, 3))
                prediction = self.detection_model.predict(img)
                if np.argmax(prediction) != 1:
                    boxes.append([y1, y2, x1, x2])
                    probabilities.append(1.0)
            except Exception as e:
                pass
        boxes = np.array(boxes)
        probabilities = np.array(probabilities)
        return boxes, probabilities

    def __filter_enclosed_boxes__(self, boxes):
        flags = np.zeros(boxes.shape[0])
        area_tuples = list()
        filtered_boxes = list()
        for box in boxes:
            y1, y2, x1, x2 = box
            area = (y2-y1)*(x2-x1)
            area_tuples.append((area, box))
        area_tuples = sorted(area_tuples, key=operator.itemgetter(0))
        for iat in range(len(area_tuples)-1):
            y11, y12, x11, x12 = area_tuples[iat][1]
            for jat in range(iat+1, len(area_tuples)):
                y21, y22, x21, x22 = area_tuples[jat][1]
                if y21<=y11 and y22>=y12 and x21<=x11 and x22>=x12:
                    flags[iat] = 1
                    break
        for ibox in range(len(flags)):
            if not flags[ibox]:
                filtered_boxes.append(area_tuples[ibox][1])
        filtered_boxes = np.array(filtered_boxes)
        return filtered_boxes        

    def detect(self, image):
        kernels = [3, 5]
        all_boxes = list()
        all_probabilities = list()
        for kernel in kernels:
            try:
                gray = image.copy()
                gray = cv2.GaussianBlur(gray, (kernel, kernel), cv2.BORDER_DEFAULT)
                regions = self.__mser__(gray)
                boxes, probabilities = self.__filter_background__(gray, regions)
                boxes, probabilities = self.__non_max_suppression__(boxes, probabilities)
                all_boxes.extend(boxes)
                all_probabilities.extend(probabilities)
            except Exception as e:
                pass
        all_boxes = np.array(all_boxes)
        all_probabilities = np.array(all_probabilities)
        all_boxes, all_probabilities = self.__non_max_suppression__(all_boxes, all_probabilities)
        all_boxes = self.__filter_enclosed_boxes__(all_boxes)
        return all_boxes
    
    def classify(self, image, boxes, model):
        gaussian_kernels = [1, 3]
        box_dicts = list()
        for box in tqdm(boxes):
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prediction_dicts = list()
                for kernel in gaussian_kernels:
                    gray = cv2.GaussianBlur(gray, (kernel, kernel), cv2.BORDER_DEFAULT)
                    y1, y2, x1, x2 = box
                    img  = gray[y1: y2, x1: x2]
                    if img.shape[0] == 0 or img.shape[1] == 0:
                        continue
                    img = cv2.resize(img, (32, 32))
                    img = img/255.
                    img = np.reshape(img, (1, 32, 32, 1))
                    prediction = model.predict(img)
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)*100
                    prediction_dicts.append({str(digit): confidence})
                print(prediction_dicts)
                if prediction_dicts[0][list(prediction_dicts[0].keys())[0]] > prediction_dicts[1][list(prediction_dicts[1].keys())[0]]:
                    prediction_dict = prediction_dicts[0]
                else:
                    prediction_dict = prediction_dicts[1]
                if prediction_dict[list(prediction_dict.keys())[0]] > 80.0:
                    if list(prediction_dict.keys())[0] != "10":
                        box_dicts.append({list(prediction_dict.keys())[0] : box})
            except:
                print("exception..........")
        return box_dicts