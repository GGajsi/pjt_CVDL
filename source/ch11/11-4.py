import os

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

img=Image.open(os.path.join(os.path.dirname(__file__), 'BSDS_361010.jpg'))

feature_extractor=DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model=DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

inputs=feature_extractor(img,return_tensors='pt')
with torch.no_grad():
    res=model(**inputs)

import numpy as np
import cv2 as cv

colors=np.random.uniform(0,255,size=(100,3))	# 100개 색으로 트랙 구분
im=cv.cvtColor(np.array(img),cv.COLOR_RGB2BGR)
width,height=img.size
for i in range(res.logits.shape[1]):
    predicted_label=res.logits[0,i].argmax(-1).item()
    if predicted_label!=91:
        name=model.config.id2label[predicted_label]
        prob='{:.2f}'.format(res.logits[0,i].softmax(dim=0)[predicted_label].item())
        box=res.pred_boxes[0,i]
        cx,cy=int(width*box[0].item()),int(height*box[1].item())
        w,h=int(width*box[2].item()),int(height*box[3].item())
        color=colors[predicted_label % len(colors)]
        cv.rectangle(im,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),color,2)
        cv.putText(im,name+': '+prob,(cx-w//2,max(cy-h//2-5,10)),cv.FONT_HERSHEY_SIMPLEX,0.6,color,1)

cv.imshow('DETR',im)
cv.waitKey()       
cv.destroyAllWindows()