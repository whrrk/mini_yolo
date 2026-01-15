import tensorflow as tf
from tensorflow.keras import layers
from model import build_yolov1
from loss import yolo_v1_loss
from dataset_tf import make_dataset

## 訓練するスクリプト
ROOT = "data"
IMG_SIZE = 320
##YOLOがイメージを数セルで分けてから見ているか
GRID_SIZE = 14
##1つのグリッドセルが予測するバウンディングボックス候補の数
##画像の中に物体がいくらぐらいいるのかによる
BBOX_COUNT = 1

# african-wildlifeは 普通 4クラス(仮定).  yaml의 ncが正解.
CLASS_COUNT = 4

BATCH = 4  # OOM 発生時 4/2で
EPOCHS = 2

def pack_y_in_dataset(x, y):
    y_p = tf.concat([y["box"], y["obj"], y["cls"]], axis=-1)  # (bs,7,7,9)
    return x, y_p

train_ds = make_dataset(ROOT, "train", IMG_SIZE, GRID_SIZE, CLASS_COUNT, batch=BATCH, shuffle=True)
val_ds   = make_dataset(ROOT, "val",   IMG_SIZE, GRID_SIZE, CLASS_COUNT, batch=BATCH, shuffle=False)
#test_ds = make_dataset(ROOT, "test", IMG_SIZE, GRID_SIZE, CLASS_COUNT, batch=BATCH, shuffle=False)

train_ds = train_ds.map(pack_y_in_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(pack_y_in_dataset,   num_parallel_calls=tf.data.AUTOTUNE)
# for x, y in train_ds.take(1):
#     print("obj mean:", float(tf.reduce_mean(y["obj"])), "obj max:", float(tf.reduce_max(y["obj"])))

model = build_yolov1(input_shape=(IMG_SIZE, IMG_SIZE, 3), grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT, backbone_trainable=False)

outs = model.output  # dict: box/obj/cls

cls = outs["cls"]  # (None,7,7,4)

if BBOX_COUNT == 1:
    # (None,7,7,1,4)->(None,7,7,4), (None,7,7,1,1)->(None,7,7,1)
    box = layers.Reshape((GRID_SIZE, GRID_SIZE, 4))(outs["box"])
    obj = layers.Reshape((GRID_SIZE, GRID_SIZE, 1))(outs["obj"])
    y_pred_p = layers.Concatenate(axis=-1)([box, obj, cls])  # (None,7,7,9)
else:
    # B個のbbox/confを全部使って学習する
    bbox_conf = layers.Concatenate(axis=-1)([outs["box"], outs["obj"]])  # (None,S,S,B,5)
    bbox_conf = layers.Reshape((GRID_SIZE, GRID_SIZE, BBOX_COUNT * 5))(bbox_conf)
    y_pred_p = layers.Concatenate(axis=-1)([bbox_conf, cls])  # (None,S,S,B*5+C)

model = tf.keras.Model(inputs=model.input, outputs=y_pred_p)
##YOLOは座標回帰と分類が混在するため、勾配スケールの違いに強いAdamが安定して学習
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Keras compile: lossに dict入力をため、 カスタム loss wrapper 使用
## loss_fn関数をtensorflow演算グラフに変換してGPUが直接実行できるようにコンパイル
@tf.function
def loss_fn(y_true, y_pred):
    return yolo_v1_loss(y_true, y_pred, grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT)

model.compile(optimizer=optimizer, loss=loss_fn)

# チェックポイント
check_point = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/mini_yolo.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

# for x,y in train_ds.take(1): print(tf.reduce_mean(y[...,4:5]), tf.reduce_max(y[...,4:5]))

# print("trainable vars:", len(model.trainable_variables))
# print("val cardinality:", tf.data.experimental.cardinality(val_ds).numpy())
# for x, y in val_ds.take(1):
#     print("val obj mean/max:", float(tf.reduce_mean(y[...,4:5])), float(tf.reduce_max(y[...,4:5])))

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[check_point])
