import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from region_proposal_classifier import Region_proposal_classifier

data_dir = "test_/"
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
thickness = 1

print("loading models")
detection_model = keras.models.load_model("models/detection_model.h5", compile=False)
classification_model = keras.models.load_model("models/classification_model.h5", compile=False)
print("models loaded")

region_proposer_classifier = Region_proposal_classifier(detection_model,classification_model)

for index, image in enumerate(os.listdir(data_dir)):
    print(image)    
    image = cv2.imread(os.path.join(data_dir, image))
    print("proposing regions...")
    boxes = region_proposer_classifier.detect(image)
    print("classifying digits...")
    box_dicts = region_proposer_classifier.classify(image, boxes, classification_model)
    for box_dict in box_dicts:
        label, box = list(box_dict.items())[0]
        box = np.array(box)
        y1, y2, x1, x2 = box
        cv2.rectangle(image, (x1, y2), (x2, y1), (255, 0, 0), 1)
        cv2.putText(image, label, (x1-5, y1-2), font, fontscale,  (255, 0, 0), thickness, cv2.LINE_AA)
    print("writing the results")
    cv2.imwrite("results/test"+str(index)+".png", image)
    # break