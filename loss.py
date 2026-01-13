import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none")

def box_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def obj_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def cls_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

##tensor  = テンソルとは、多次元の数値配列で、画像やYOLOの出力もすべてテンソル
def xywh_to_xyxy(xywh):
    # xywh: (...,4) where x,y center
    x, y, w, h = tf.split(xywh, 4, axis=-1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return tf.concat([x1,y1,x2,y2], axis=-1)

def bbox_iou_xywh(box1, box2, eps=1e-7):
    # box1: (...,4), box2: (...,4) broadcast 可能
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

def cell_xywh_to_abs_xywh(xywh, grid_size):
    # xywh: (...,4) with x,y in cell coords (0~1), w,h in image coords (0~1)
    x, y, w, h = tf.split(xywh, 4, axis=-1)
    grid_y = tf.range(grid_size, dtype=xywh.dtype)
    grid_x = tf.range(grid_size, dtype=xywh.dtype)
    yy, xx = tf.meshgrid(grid_y, grid_x, indexing="ij")  # (S,S)
    xx = tf.reshape(xx, (1, grid_size, grid_size, 1, 1))
    yy = tf.reshape(yy, (1, grid_size, grid_size, 1, 1))
    x_abs = (xx + x) / grid_size
    y_abs = (yy + y) / grid_size
    return tf.concat([x_abs, y_abs, w, h], axis=-1)

# y_true(正解)(data_setが作った), y_pred(予測)(modelで出力)を比較してどれぐらいズレあるかをスカラで作る
def yolo_v1_loss(y_true, y_pred, grid_size=7, bbox_count=2, class_count=20,
                 lambda_coord=5.0, lambda_noobj=0.05):
    """
    y_true dict 形態勧奨:
      y_true["box"] : (bs,S,S,4)  xywh (cell-relative, 0~1)
      y_true["obj"] bbox_count: (bs,S,S,1)  0/bbox_count
      y_true["cls"] : (bs,S,S,C)  one-hot
    """

    true_box = y_true[..., 0:4]
    true_obj = y_true[..., 4:5]
    true_cls = y_true[..., 5:5+class_count]

    # pred parsing
    pred_bbox_conf = y_pred[..., :bbox_count * 5]
    pred_cls = y_pred[..., bbox_count * 5:bbox_count * 5 + class_count]  # (bs,S,S,C)
    pred_bbox_conf = tf.reshape(
        pred_bbox_conf, (-1, grid_size, grid_size, bbox_count, 5)
    )
    pred_box = pred_bbox_conf[..., 0:4]   # (bs,S,S,B,4)
    pred_conf = pred_bbox_conf[..., 4:5]  # (bs,S,S,B,1)

    # 責任 bbox 選択: IoUが 一番大きい bbox index
    true_box_exp = tf.expand_dims(true_box, axis=3)  # (bs,S,S,1,4)
    pred_abs = cell_xywh_to_abs_xywh(pred_box, grid_size)
    true_abs = cell_xywh_to_abs_xywh(true_box_exp, grid_size)
    iou = bbox_iou_xywh(pred_abs, true_abs)  # (bs,S,S,B,1)
    iou = tf.squeeze(iou, axis=-1)          # (bs,S,S,B)
    best_idx = tf.argmax(iou, axis=-1)      # (bs,S,S)
    best_mask = tf.one_hot(best_idx, bbox_count, dtype=pred_box.dtype)  # (bs,S,S,B)
    best_mask = tf.expand_dims(best_mask, axis=-1)  # (bs,S,S,B,1)

    true_obj_exp = tf.expand_dims(true_obj, axis=3)  # (bs,S,S,1,1)
    true_obj_tiled = tf.tile(true_obj_exp, [1, 1, 1, bbox_count, 1])
    responsible = best_mask * true_obj_tiled

    obj_bce_pos = bce(tf.ones_like(pred_conf), pred_conf)
    obj_bce_neg = bce(tf.zeros_like(pred_conf), pred_conf)
    obj_bce_pos = tf.expand_dims(obj_bce_pos, axis=-1)
    obj_bce_neg = tf.expand_dims(obj_bce_neg, axis=-1)
    obj_loss_pos = tf.reduce_sum(responsible * obj_bce_pos)
    obj_loss_neg = tf.reduce_sum((1.0 - responsible) * obj_bce_neg)
    obj_loss = obj_loss_pos + lambda_noobj * obj_loss_neg

    # responsible: 個体ある CELLで best bboxだけ 1　、 AND処理する
    # obj_maskは「そのセルに物体があるか」、best_maskは「どのボックスが代表か」を示し、その論理積としてresponsible boxを決めています
    # ----- coord loss (responsible bbox only) -----
    # YOLOv1は w,hに sqrt 適用
    # sqrtは小さいバウンディングボックスの誤差をより強く学習させるために使います。

    pred_xy = pred_box[..., 0:2] #YOLOが予測したバウンディングボックスの中心座標(x,y)だけを取り出し
    pred_wh = pred_box[..., 2:4]
    true_xy = true_box_exp[..., 0:2]
    true_wh = true_box_exp[..., 2:4]

    coord_loss_xy = tf.reduce_sum(responsible * tf.square(pred_xy - true_xy))
    coord_loss_wh = tf.reduce_sum(
        responsible * tf.square(tf.sqrt(pred_wh + 1e-9) - tf.sqrt(true_wh + 1e-9))
    )
    coord_loss = lambda_coord * (coord_loss_xy + coord_loss_wh)

    # ----- noobj loss -----
    # 個体はあるけど 責任ではない bboxは noobjで(簡単化)
    # CELLに 個体が ない時も noobj 含めると:
    noobj_loss = 0.0

    # ----- class loss (個体あるCELLだけ) -----
    # pred_cls: (bs,S,S,C)
    class_loss = tf.reduce_sum(true_obj * tf.reduce_sum(tf.square(pred_cls - true_cls), axis=-1, keepdims=True))

    ##ロースの正規化
    total = coord_loss + obj_loss + noobj_loss + class_loss
    denom = tf.cast(tf.shape(y_true)[0] * grid_size * grid_size, total.dtype)
    denom = tf.maximum(denom, 1.0)
    return total / denom
