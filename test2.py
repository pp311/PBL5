import os  
import torch
import cv2 
import yaml
import time
import pyrebase
import numpy as np

            # load model
# model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5\runs\train\exp7\weights\best.pt', source='local') 
# model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path='yolov5s.pt', source='local') 
model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp19\last.pt', source='local') 
# model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp21\last.pt', source='local')
            # Images
# imgs = cv2.imread("image0.jpg")  # Replace with your image file path
imgs = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5\data\images\pho.jpg")
# Inference
dectections = model(imgs)

# results
results = dectections.pandas().xyxy[0].to_dict(orient="records")
# x = np.array(results)
# print(x)

# filter
for result in results:
    confidence = result['confidence']
    print(confidence)
    name = result['name']
    clas = result['class']
    print('vo day roi')
    if clas == 0:
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        # print(x1,y1,x2,y2)
    
        #draw rectangle
        cv2.rectangle(imgs,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(imgs, name, (x1+3,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
# show picture
# cv2.imwrite(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh',imgs)
print('toi day')
cv2.imshow('img',imgs)

cv2.waitKey(0)

