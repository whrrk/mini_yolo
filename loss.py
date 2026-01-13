import tensorflow as tf

def xywh_to_xyxy(xywh):
    # xywh: (...,4) where x,y center
    x, y, w, h = tf.split(xywh, 4, axis=-1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return tf.concat([x1,y1,x2,y2], axis=-1)

def bbox_iou_xywh(box1, box2, eps=1e-7):
    # box1: (...,4), box2: (...,4) broadcast 가능
    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)
    b1x1,b1y1,b1x2,b1y2 = tf.split(b1, 4, axis=-1)
    b2x1,b2y1,b2x2,b2y2 = tf.split(b2, 4, axis=-1)

    inter_x1 = tf.maximum(b1x1, b2x1)
    inter_y1 = tf.maximum(b1y1, b2y1)
    inter_x2 = tf.minimum(b1x2, b2x2)
    inter_y2 = tf.minimum(b1y2, b2y2)
    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    area1 = tf.maximum(b1x2 - b1x1, 0.0) * tf.maximum(b1y2 - b1y1, 0.0)
    area2 = tf.maximum(b2x2 - b2x1, 0.0) * tf.maximum(b2y2 - b2y1, 0.0)
    union = area1 + area2 - inter
    return inter / (union + eps)

def yolo_v1_loss(y_true, y_pred, grid_size=7, bbox_count=2,
                 lambda_coord=5.0, lambda_noobj=0.5):
    """
    y_true dict 形態勧奨:
      y_true["box"] : (bs,S,S,4)  xywh (cell-relative, 0~1)
      y_true["obj"] : (bs,S,S,1)  0/1
      y_true["cls"] : (bs,S,S,C)  one-hot
    """

    true_box = y_true["box"]
    true_obj = y_true["obj"]
    true_cls = y_true["cls"]

    # pred parsing
    pred_bbox_conf = y_pred[..., :bbox_count*5]                        # (bs,S,S,bbox_count*5)
    pred_cls = y_pred[..., bbox_count*5:]                               # (bs,S,S,C)
    pred_bbox_conf = tf.reshape(pred_bbox_conf, (-1, grid_size, grid_size, bbox_count, 5))
    pred_xywh = pred_bbox_conf[..., 0:4]                       # (bs,S,S,bbox_count,4)
    pred_conf = pred_bbox_conf[..., 4:5]                       # (bs,S,S,bbox_count,1)

    # 責任 bbox 選択: IoUが 一番大きい bbox index
    # true_boxを B個に broadcast: (bs,S,S,1,4)
    true_box_exp = tf.expand_dims(true_box, axis=3)
    iou = bbox_iou_xywh(pred_xywh, true_box_exp)               # (bs,S,S,bbox_count,1) 又は (bs,S,S,bbox_count,1)
    iou = tf.squeeze(iou, axis=-1)                             # (bs,S,S,bbox_countB)

    best_idx = tf.argmax(iou, axis=-1)                         # (bs,S,S)
    best_mask = tf.one_hot(best_idx, depth=bbox_count, dtype=tf.float32) # (bs,S,S,bbox_count)
    best_mask = tf.expand_dims(best_mask, axis=-1)             # (bs,S,S,bbox_count,1)

    # object mask 拡張
    obj_mask = tf.expand_dims(true_obj, axis=3)                # (bs,S,S,1,1)
    obj_mask = tf.tile(obj_mask, [1,1,1,bbox_count,1])                  # (bs,S,S,bbox_count,1)

    # responsible: 個体ある CELLで best bboxだけ 1
    resp_mask = obj_mask * best_mask                           # (bs,S,S,bbox_count,1)

    # ----- coord loss (responsible bbox only) -----
    # YOLOv1은 w,hに sqrt 適用
    pred_xy = pred_xywh[..., 0:2]
    pred_wh = pred_xywh[..., 2:4]
    true_xy = tf.expand_dims(true_box[..., 0:2], axis=3)
    true_wh = tf.expand_dims(true_box[..., 2:4], axis=3)

    coord_loss_xy = tf.reduce_sum(resp_mask * tf.square(pred_xy - true_xy))
    coord_loss_wh = tf.reduce_sum(resp_mask * tf.square(tf.sqrt(pred_wh + 1e-9) - tf.sqrt(true_wh + 1e-9)))
    coord_loss = lambda_coord * (coord_loss_xy + coord_loss_wh)

    # ----- objectness loss -----
    # responsible bboxも confは IoU targetを 使うこともある
    iou_target = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # (bs,S,S,1)
    iou_target = tf.expand_dims(iou_target, axis=3)                    # (bs,S,S,1,1)
    iou_target = tf.tile(iou_target, [1,1,1,bbox_count,1])                      # (bs,S,S,bbox_count,1)
    obj_loss = tf.reduce_sum(resp_mask * tf.square(pred_conf - iou_target))

    # ----- noobj loss -----
    noobj_mask = (1.0 - resp_mask) * obj_mask  # 個体はあるけど 責任ではない bboxは noobjで(簡単化)
    # CELLに 個体が ない時も noobj 含めると:
    noobj_mask = (1.0 - resp_mask) * tf.expand_dims(tf.ones_like(true_obj), 3)  # 全体 bbox 対象
    noobj_loss = lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(pred_conf - 0.0))

    # ----- class loss (個体あるCELLだけ) -----
    # pred_cls: (bs,S,S,C)
    class_loss = tf.reduce_sum(true_obj * tf.reduce_sum(tf.square(pred_cls - true_cls), axis=-1, keepdims=True))

    total = coord_loss + obj_loss + noobj_loss + class_loss
    return total
