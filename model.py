import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_yolov1(input_shape=(320, 320, 3), #モデルが受け取る画像の解像度とRGBチャネル数を指定
                 grid_size=7, 
                 bbox_count=2, 
                 class_count=20, 
                 backbone_trainable=False #事前学習済みの特徴抽出器を固定して、検出ヘッドだけを学習する設定 
                 ):
    
    inp = keras.Input(shape=input_shape)

    # Backbone (例: MobileNetV2)
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=inp
    )

    base.trainable = backbone_trainable

    # backbone 出力: (H/32, W/32, ch)  → input=320なら 約 (10,10,1280)
    x = base.output

    # Detection head: grid_sizexgrid_sizeに 合わせるため Conv + Resize/Pooling
    # 簡単に (grid_size,grid_size)に リサイズ
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 1, padding="same", activation="relu")(x)

    # (grid_size,grid_size,channels)に 強制 変換
    x = tf.image.resize(x, (grid_size, grid_size), method="bilinear")

    # 最終 予測 logits
    pred = layers.Conv2D(bbox_count*5 + class_count, 1, padding="same")(x)  # (grid_size,grid_size,bbox_count*5+class_count)

    # ====== 出力 分離 + activation ======
    # pred[..., :bbox_count*5] -> bbox + conf
    # pred[..., bbox_count*5:] -> class logits
    bbox_conf = pred[..., :bbox_count*5]
    cls_logits = pred[..., bbox_count*5:]

    # bbox_conf reshape: (grid_size,grid_size,bbox_count,5)
    bbox_conf = tf.reshape(bbox_conf, (-1, grid_size, grid_size, bbox_count, 5))

    # x,y -> sigmoid
    xy = tf.sigmoid(bbox_conf[..., 0:2])
    # w,h -> softplus (正数 安定)
    wh = tf.nn.softplus(bbox_conf[..., 2:4])
    # conf -> sigmoid
    conf = tf.sigmoid(bbox_conf[..., 4:5])

    # class -> softmax
    cls = tf.nn.softmax(cls_logits, axis=-1)

    # 足す: (grid_size,grid_size,bbox_count,5) + (grid_size,grid_size,class_count) を　また (grid_size,grid_size,bbox_count*5+class_count) 形態に
    bbox_conf_act = tf.concat([xy, wh, conf], axis=-1)               # (grid_size,grid_size,bbox_count,5)
    bbox_conf_act = tf.reshape(bbox_conf_act, (-1, grid_size, grid_size, bbox_count*5))       # (grid_size,grid_size,bbox_count*5)
    out = tf.concat([bbox_conf_act, cls], axis=-1)                   # (grid_size,grid_size,bbox_count*5+class_count)

    return keras.Model(inp, out)
