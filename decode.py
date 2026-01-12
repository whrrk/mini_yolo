import tensorflow as tf

def decode_yolov1_output(y_pred, S=7, B=2, C=20, conf_thres=0.25, iou_thres=0.5):
    """
    y_pred: (1,S,S,B*5+C) 또는 (S,S,B*5+C)
    return: boxes (N,4) in normalized xyxy, scores (N,), class_ids (N,)
    """
    if len(y_pred.shape) == 4:
        y_pred = y_pred[0]  # (S,S,...)  バッチ1 仮定

    pred_bbox_conf = y_pred[..., :B*5]
    pred_cls = y_pred[..., B*5:]  # (S,S,C)

    pred_bbox_conf = tf.reshape(pred_bbox_conf, (S, S, B, 5))
    xy = pred_bbox_conf[..., 0:2]         # (S,S,B,2) 0~1
    wh = pred_bbox_conf[..., 2:4]         # (S,S,B,2) positive
    conf = pred_bbox_conf[..., 4]         # (S,S,B)

    # grid offsets
    grid_y = tf.range(S, dtype=tf.float32)
    grid_x = tf.range(S, dtype=tf.float32)
    yy, xx = tf.meshgrid(grid_y, grid_x, indexing="ij")  # (S,S)
    xx = tf.expand_dims(xx, axis=-1)  # (S,S,1)
    yy = tf.expand_dims(yy, axis=-1)  # (S,S,1)

    # abs center (normalized 0~1)
    x_abs = (xx + xy[..., 0]) / S
    y_abs = (yy + xy[..., 1]) / S
    w_abs = wh[..., 0]
    h_abs = wh[..., 1]

    # xywh -> xyxy
    x1 = x_abs - w_abs / 2.0
    y1 = y_abs - h_abs / 2.0
    x2 = x_abs + w_abs / 2.0
    y2 = y_abs + h_abs / 2.0

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)  # (S,S,B,4)

    # class scores
    cls_prob = pred_cls  # (S,S,C) already softmax
    cls_id = tf.argmax(cls_prob, axis=-1)        # (S,S)
    cls_score = tf.reduce_max(cls_prob, axis=-1) # (S,S)

    # expand to B boxes per cell: (S,S,B)
    cls_id = tf.expand_dims(cls_id, axis=-1)
    cls_id = tf.tile(cls_id, [1,1,B])
    cls_score = tf.expand_dims(cls_score, axis=-1)
    cls_score = tf.tile(cls_score, [1,1,B])

    scores = conf * cls_score  # (S,S,B)

    # flatten
    boxes_f = tf.reshape(boxes, (-1, 4))
    scores_f = tf.reshape(scores, (-1,))
    cls_f = tf.reshape(cls_id, (-1,))

    # threshold
    keep = scores_f >= conf_thres
    boxes_f = tf.boolean_mask(boxes_f, keep)
    scores_f = tf.boolean_mask(scores_f, keep)
    cls_f = tf.boolean_mask(cls_f, keep)

    if tf.shape(scores_f)[0] == 0:
        return boxes_f, scores_f, cls_f

    # NMS (class-agnostic 簡単バージョン)
    selected = tf.image.non_max_suppression(
        boxes_f, scores_f, max_output_size=300, iou_threshold=iou_thres
    )
    boxes_n = tf.gather(boxes_f, selected)
    scores_n = tf.gather(scores_f, selected)
    cls_n = tf.gather(cls_f, selected)
    return boxes_n, scores_n, cls_n
