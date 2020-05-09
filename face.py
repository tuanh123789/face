import cv2
import numpy as np
import os
from PIL import Image
import pickle
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"images")
nhandien = cv2.face.LBPHFaceRecognizer_create()
labels={1}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
current_id=0
labels_id={}
x_train=[]
y_labels=[]
for root,dir,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            #print(label,path)
            if label in labels_id:
                pass
            else:
                labels_id[label]=current_id
                current_id += 1
            id_=labels_id[label]
            #print(labels_id)
            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image)
            #print(image_array)
            faces1=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in faces1:
                roi=image_array[y:y+h,x:x+h]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(labels_id,f)
nhandien.train(x_train,np.array(y_labels))

cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=nhandien.predict(roi_gray)
        if conf>=45 and conf<= 85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,0,0)
            doday=3
            cv2.putText(frame,name,(x,y),font,1,color,doday,cv2.LINE_AA)
        color=(0,255,0)
        stroke=5
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
