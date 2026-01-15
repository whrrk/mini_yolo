import tensorflow as tf

def decode_yolov1_output(
    y_pred,
    grid_size=7,
    bbox_count=1,
    conf_thres=0.25,
    iou_thres=0.5,
    obj_thres=0.1,
    class_agnostic_nms=False,
    return_pre_nms=False,
    max_output_size=300,
):
    """
    y_pred: (1,S,S,B*5+class_count) 又は (S,S,B*5+class_count)
    return: boxes (N,4) in normalized xyxy, scores (N,), class_ids (N,)
    """
    if isinstance(y_pred, dict):
        box = tf.squeeze(y_pred["box"], axis=3)  # (bs,S,S,4)
        obj = tf.squeeze(y_pred["obj"], axis=3)  # (bs,S,S,1)
        cls = y_pred["cls"]                      # (bs,S,S,C)
        y_pred = tf.concat([box, obj, cls], axis=-1)  # (bs,S,S,5+C)
        bbox_count = 1  # 今のモデル出力は実質B=1

    if len(y_pred.shape) == 4:
        if y_pred.shape[0] == 1:
            y_pred = y_pred[0]

    pred_bbox_conf = y_pred[..., :bbox_count*5]
    pred_cls = y_pred[..., bbox_count*5:]  # (S,S,class_count)

    pred_bbox_conf = tf.reshape(pred_bbox_conf, (grid_size, grid_size, bbox_count, 5))
    xy = pred_bbox_conf[..., 0:2]         # (S,S,bbox_count,2) 0~1
    wh = pred_bbox_conf[..., 2:4]         # (S,S,bbox_count,2) positive
    conf = pred_bbox_conf[..., 4]         # (S,S,bbox_count)

    # grid offsets
    grid_y = tf.range(grid_size, dtype=tf.float32)
    grid_x = tf.range(grid_size, dtype=tf.float32)
    yy, xx = tf.meshgrid(grid_y, grid_x, indexing="ij")  # (S,S)
    xx = tf.expand_dims(xx, axis=-1)  # (S,S,1)
    yy = tf.expand_dims(yy, axis=-1)  # (S,S,1)

    # abs center (normalized 0~1)
    x_abs = (xx + xy[..., 0]) / grid_size
    y_abs = (yy + xy[..., 1]) / grid_size
    w_abs = wh[..., 0]
    h_abs = wh[..., 1]

    # xywh -> xyxy
    x1 = x_abs - w_abs / 2.0
    y1 = y_abs - h_abs / 2.0
    x2 = x_abs + w_abs / 2.0
    y2 = y_abs + h_abs / 2.0

    boxes_xyxy = tf.stack([x1, y1, x2, y2], axis=-1)  # (S,S,bbox_count,4)
    boxes_nms = tf.stack([y1, x1, y2, x2], axis=-1)   # (S,S,bbox_count,4)

    # class scores
    cls_prob = pred_cls  # (S,S,class_count) already softmax
    cls_id = tf.argmax(cls_prob, axis=-1)        # (S,S)
    cls_score = tf.reduce_max(cls_prob, axis=-1) # (S,S)

    # expand to bbox_count boxes per cell: (S,S,Bbbox_count)
    cls_id = tf.expand_dims(cls_id, axis=-1)
    cls_id = tf.tile(cls_id, [1, 1, bbox_count])
    cls_score = tf.expand_dims(cls_score, axis=-1)
    cls_score = tf.tile(cls_score, [1, 1, bbox_count])

    scores = conf * cls_score  # (S,S,bbox_count)

    # keep only best bbox per cell to reduce duplicates
    best_b = tf.argmax(scores, axis=-1, output_type=tf.int32)  # (S,S)
    best_mask = tf.one_hot(best_b, bbox_count, dtype=scores.dtype)  # (S,S,B)

    best_mask_4 = tf.expand_dims(best_mask, axis=-1)  # (S,S,B,1)
    boxes_xyxy = tf.reduce_sum(boxes_xyxy * best_mask_4, axis=2)     # (S,S,4)
    boxes_nms = tf.reduce_sum(boxes_nms * best_mask_4, axis=2)        # (S,S,4)
    conf = tf.reduce_sum(conf * best_mask, axis=2)         # (S,S)
    scores = tf.reduce_sum(scores * best_mask, axis=2)     # (S,S)
    cls_id = tf.reduce_sum(cls_id * tf.cast(best_mask, cls_id.dtype), axis=2)  # (S,S)

    boxes_xyxy = tf.expand_dims(boxes_xyxy, axis=2)   # (S,S,1,4)
    boxes_nms = tf.expand_dims(boxes_nms, axis=2)     # (S,S,1,4)
    conf = tf.expand_dims(conf, axis=2)     # (S,S,1)
    scores = tf.expand_dims(scores, axis=2) # (S,S,1)
    cls_id = tf.expand_dims(cls_id, axis=2) # (S,S,1)

    # flatten
    boxes_f = tf.reshape(boxes_xyxy, (-1, 4))
    boxes_nms_f = tf.reshape(boxes_nms, (-1, 4))
    conf_f = tf.reshape(conf, (-1,))
    scores_f = tf.reshape(scores, (-1,))
    cls_f = tf.reshape(cls_id, (-1,))

    # threshold
    boxes_all = boxes_f
    scores_all = scores_f
    conf_all = conf_f
    cls_all = cls_f

    keep = tf.logical_and(scores_f >= conf_thres, conf_f >= obj_thres)
    boxes_f = tf.boolean_mask(boxes_f, keep)
    boxes_nms_f = tf.boolean_mask(boxes_nms_f, keep)
    scores_f = tf.boolean_mask(scores_f, keep)
    cls_f = tf.boolean_mask(cls_f, keep)

    if tf.shape(scores_f)[0] == 0:
        k = tf.minimum(10, tf.shape(scores_all)[0])
        if k == 0:
            return boxes_f, scores_f, cls_f
        topk = tf.math.top_k(scores_all, k=k).indices
        boxes_f = tf.gather(boxes_all, topk)
        boxes_nms_f = tf.gather(boxes_nms_f, topk)
        scores_f = tf.gather(scores_all, topk)
        cls_f = tf.gather(cls_all, topk)

    if return_pre_nms:
        return boxes_f, scores_f, cls_f

    # NMS (class-wise or class-agnostic)
    class_count = pred_cls.shape[-1]
    if class_count is None:
        class_count = int(tf.shape(pred_cls)[-1])

    if class_agnostic_nms:
        selected = tf.image.non_max_suppression(
            boxes_nms_f, scores_f, max_output_size=max_output_size, iou_threshold=iou_thres
        )
        boxes_n = tf.gather(boxes_f, selected)
        scores_n = tf.gather(scores_f, selected)
        cls_n = tf.gather(cls_f, selected)
        return boxes_n, scores_n, cls_n

    boxes_out = []
    scores_out = []
    cls_out = []
    for cid in range(class_count):
        mask = tf.equal(cls_f, cid)
        boxes_c = tf.boolean_mask(boxes_f, mask)
        boxes_nms_c = tf.boolean_mask(boxes_nms_f, mask)
        scores_c = tf.boolean_mask(scores_f, mask)
        if tf.size(scores_c) == 0:
            continue
        selected = tf.image.non_max_suppression(
            boxes_nms_c, scores_c, max_output_size=max_output_size, iou_threshold=iou_thres
        )
        boxes_out.append(tf.gather(boxes_c, selected))
        scores_out.append(tf.gather(scores_c, selected))
        cls_out.append(tf.fill((tf.size(selected),), cid))

    if not boxes_out:
        return boxes_f, scores_f, cls_f

    boxes_n = tf.concat(boxes_out, axis=0)
    scores_n = tf.concat(scores_out, axis=0)
    cls_n = tf.concat(cls_out, axis=0)
    return boxes_n, scores_n, cls_n
