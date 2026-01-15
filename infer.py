import tensorflow as tf
import numpy as np
import cv2
from model import build_yolov1
from decode import decode_yolov1_output  # decode関数

IMG_SIZE = 320
GRID_SIZE=7; 
BBOX_COUNT=1; 
CLASS_COUNT=4
##最終スコアの下限
CONF_THRES = 0.5
##物体がある確率
OBJ_THRES = 0.5
##同じ物体とみなして消す
IOU_THRES = 0.6
DEBUG_DRAW_PRE_NMS = True
##NMS= 重なった複数の箱から高いものだけを残す
NMS_MAX_OUTPUT = 200

# クラスの名前 - yamlの名前
NAMES = ["buffalo","elephant","rhino","zebra"]

model = build_yolov1((IMG_SIZE,IMG_SIZE,3), grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT, backbone_trainable=False)
##学習中性能が良かった瞬間の神経網重みをmini_yolo.weights.h5に保存する”
model.load_weights("checkpoints/mini_yolo.weights.h5")

def preprocess(path):
    bgr = cv2.imread(path)
    h,w = bgr.shape[:2]
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)/255.0
    return bgr, img, (h,w)

bgr, img, (H,W) = preprocess("data/images/train/4 (151).jpg")

y_pred = model(img[None, ...], training=False)

if BBOX_COUNT == 1:
    box = tf.squeeze(y_pred["box"], axis=3)   # (1,S,S,4)
    obj = tf.squeeze(y_pred["obj"], axis=3)   # (1,S,S,1)
    cls = y_pred["cls"]                       # (1,S,S,C)
    y_pred_packed = tf.concat([box, obj, cls], axis=-1)  # (1,S,S,5+C)
else:
    box = y_pred["box"]  # (1,S,S,B,4)
    obj = y_pred["obj"]  # (1,S,S,B,1)
    cls = y_pred["cls"]  # (1,S,S,C)
    bbox_conf = tf.concat([box, obj], axis=-1)  # (1,S,S,B,5)
    bbox_conf = tf.reshape(
        bbox_conf, (1, GRID_SIZE, GRID_SIZE, BBOX_COUNT * 5)
    )
    y_pred_packed = tf.concat([bbox_conf, cls], axis=-1)  # (1,S,S,B*5+C)

print("pred box min/max:", float(tf.reduce_min(box)), float(tf.reduce_max(box)))
print("pred obj min/max:", float(tf.reduce_min(obj)), float(tf.reduce_max(obj)))
print("pred cls min/max:", float(tf.reduce_min(cls)), float(tf.reduce_max(cls)))

boxes, scores, cls_ids = decode_yolov1_output(
    y_pred_packed,
    grid_size=GRID_SIZE,
    bbox_count=BBOX_COUNT,
    conf_thres=CONF_THRES,
    iou_thres=IOU_THRES,
    obj_thres=OBJ_THRES,
    class_agnostic_nms=False,
    max_output_size=NMS_MAX_OUTPUT
)

if DEBUG_DRAW_PRE_NMS:
    pre_boxes, pre_scores, pre_cls_ids = decode_yolov1_output(
        y_pred_packed,
        grid_size=GRID_SIZE,
        bbox_count=BBOX_COUNT,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        obj_thres=OBJ_THRES,
        class_agnostic_nms=False,
        return_pre_nms=True,
        max_output_size=NMS_MAX_OUTPUT
    )



boxes = boxes.numpy()
scores = scores.numpy()
cls_ids = cls_ids.numpy().astype(int)

for (x1,y1,x2,y2), sc, cid in zip(boxes, scores, cls_ids):
    x1 = int(np.clip(x1*W, 0, W-1))
    y1 = int(np.clip(y1*H, 0, H-1))
    x2 = int(np.clip(x2*W, 0, W-1))
    y2 = int(np.clip(y2*H, 0, H-1))

    label = f"{NAMES[cid]} {sc:.2f}"

    (font_w, font_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
    )
    cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    # cv2.rectangle(
    #     bgr,
    #     (x1, y1 - font_h - 15),
    #     (x1 + font_w + 10, y1),
    #     (0, 255, 0),
    #     -1
    # )
    cv2.putText(
        bgr,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3
    )

if DEBUG_DRAW_PRE_NMS:
    pre_boxes = pre_boxes.numpy()
    pre_scores = pre_scores.numpy()
    pre_cls_ids = pre_cls_ids.numpy().astype(int)
    for (x1, y1, x2, y2), sc, cid in zip(pre_boxes, pre_scores, pre_cls_ids):
        x1 = int(np.clip(x1 * W, 0, W - 1))
        y1 = int(np.clip(y1 * H, 0, H - 1))
        x2 = int(np.clip(x2 * W, 0, W - 1))
        y2 = int(np.clip(y2 * H, 0, H - 1))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imwrite("pred_result.jpg", bgr)

print("boxes shape:", boxes.shape)
print("scores shape:", scores.shape)
print("max score:", float(tf.reduce_max(scores)) if tf.size(scores) > 0 else "no scores")
print("saved: pred_result.jpg")
