import os  
import torch
import cv2 
import yaml
import time
import pyrebase
import numpy as np

# load model 
# model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path='yolov5s.pt', source='local') 
model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp19\last.pt', source='local') 
# model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp21\last.pt', source='local')
imgs = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5\data\images\zidane.jpg")
            # Inference
results = model(imgs)
confidences = results.pred[0].detach().cpu().numpy()[:, 4]
            # print('do tin cay la: ')
            # print(confidences[0])
person_class_id = 0
if person_class_id in results.pred[0][:, -1].tolist() and confidences[0]>0.5:
    print('co nguoi')
else: 
    print('khong co nguoi')
# # Results
results.print()
results.save()  # or .show()

            