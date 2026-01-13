import tensorflow as tf
import numpy as np
import cv2
from model import build_yolov1
from decode import decode_yolov1_output  # decode関数

IMG_SIZE = 320
GRID_SIZE=7; 
BBOX_COUNT=2; 
CLASS_COUNT=4

# クラスの名前 - yamlの名前
NAMES = ["buffalo","elephant","rhino","zebra"]

model = build_yolov1((IMG_SIZE,IMG_SIZE,3), grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT, backbone_trainable=False)
model.load_weights("checkpoints/mini_yolo.weights.h5")

def preprocess(path):
    bgr = cv2.imread(path)
    h,w = bgr.shape[:2]
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)/255.0
    return bgr, img, (h,w)

bgr, img, (H,W) = preprocess("african-wildlife/images/test/000000.jpg")

y_pred = model(img[None, ...], training=False).numpy()
boxes, scores, cls_ids = decode_yolov1_output(y_pred, grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, conf_thres=0.25, iou_thres=0.5)

# normalized xyxy -> pixel
boxes = boxes.numpy()
scores = scores.numpy()
cls_ids = cls_ids.numpy().astype(int)

for (x1,y1,x2,y2), sc, cid in zip(boxes, scores, cls_ids):
    x1 = int(np.clip(x1*W, 0, W-1))
    y1 = int(np.clip(y1*H, 0, H-1))
    x2 = int(np.clip(x2*W, 0, W-1))
    y2 = int(np.clip(y2*H, 0, H-1))
    cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(bgr, f"{NAMES[cid]} {sc:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("pred_result.jpg", bgr)
print("saved: pred_result.jpg")
