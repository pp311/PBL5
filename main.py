import os  
import torch
import cv2 
import yaml
import time
import pyrebase
import numpy as np
config = {
        "apiKey":'AIzaSyDG_m77TMMnIzcQnlAm3GIDttK91lXqHdw',
        "authDomain":'pbl5-c5b07.firebaseapp.com',
        "databaseURL":'https://pbl5-c5b07-default-rtdb.firebaseio.com/',
        "storageBucket":'pbl5-c5b07.appspot.com'
        }
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database=firebase.database()

while True:
    val_mode_alert = database.child("devices/tdt@gmail,com/alert_mode").get().val()
    # print("da vo day 2")
    if val_mode_alert == "on":
        # print("da vo day 1")
        val_is_alert = database.child("devices/tdt@gmail,com/is_alert").get().val()
        if val_is_alert == False:
            # print("da vo day")
            #download anh
            storage.child("images/image0.jpg").download("image0.jpg", filename="image0.jpg")

            # load model
            model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp19\last.pt', source='local') 
            # model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path='yolov5s.pt', source='local') 
            # model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp7\last.pt', source='local') 
            # model = torch.hub.load(r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5', 'custom', path=r'C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\weights_exp21\last.pt', source='local') 
            # Images
            imgs = cv2.imread("image0.jpg")  # Replace with your image file path
            # imgs = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\PBL5_AI_HinhAnh\yolov5\data\images\pho.jpg")

            dectections = model(imgs)
            # results
            results = dectections.pandas().xyxy[0].to_dict(orient="records")
            # x = np.array(results)
            # print(x)
            dem = 0
            # filter
            for result in results:
                confidence = result['confidence']
                # print(confidence)
                name = result['name']
                clas = result['class']
                dem = dem + 1
                if clas == 0:
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    # print(x1,y1,x2,y2)
                    #draw rectangle
                    cv2.rectangle(imgs,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(imgs, name, (x1+3,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
            if (dem > 0):
                storage.child("images/person.jpg").put("image0.jpg")
                database.child("devices/tdt@gmail,com/is_alert").set(True)
            else :
                print('khong co nguoi')



            