import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_yolov1(input_shape=(320, 320, 3), S=7, B=2, C=20, backbone_trainable=False):
    inp = keras.Input(shape=input_shape)

    # Backbone (예: MobileNetV2)
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=inp
    )
    base.trainable = backbone_trainable

    # backbone 출력: (H/32, W/32, ch)  → input=320이면 대략 (10,10,1280)
    x = base.output

    # Detection head: SxS로 맞추기 위해 Conv + Resize/Pooling
    # 여기서는 간단하게 (S,S)로 리사이즈
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 1, padding="same", activation="relu")(x)

    # (S,S,channels)로 강제 변환
    x = tf.image.resize(x, (S, S), method="bilinear")

    # 최종 예측 logits
    pred = layers.Conv2D(B*5 + C, 1, padding="same")(x)  # (S,S,B*5+C)

    # ====== 출력 분리 + activation ======
    # pred[..., :B*5] -> bbox + conf
    # pred[..., B*5:] -> class logits
    bbox_conf = pred[..., :B*5]
    cls_logits = pred[..., B*5:]

    # bbox_conf reshape: (S,S,B,5)
    bbox_conf = tf.reshape(bbox_conf, (-1, S, S, B, 5))

    # x,y -> sigmoid
    xy = tf.sigmoid(bbox_conf[..., 0:2])
    # w,h -> softplus (양수 안정)
    wh = tf.nn.softplus(bbox_conf[..., 2:4])
    # conf -> sigmoid
    conf = tf.sigmoid(bbox_conf[..., 4:5])

    # class -> softmax
    cls = tf.nn.softmax(cls_logits, axis=-1)

    # 합치기: (S,S,B,5) + (S,S,C) 를 다시 (S,S,B*5+C) 형태로
    bbox_conf_act = tf.concat([xy, wh, conf], axis=-1)               # (S,S,B,5)
    bbox_conf_act = tf.reshape(bbox_conf_act, (-1, S, S, B*5))       # (S,S,B*5)
    out = tf.concat([bbox_conf_act, cls], axis=-1)                   # (S,S,B*5+C)

    return keras.Model(inp, out)
