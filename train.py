import tensorflow as tf
from model import build_yolov1
from loss import yolo_v1_loss
from dataset_tf import make_dataset

ROOT = "data"
IMG_SIZE = 320
##イメージを数セルで分けてから見ているか
GRID_SIZE = 7
##1つのグリッドセルが予測するバウンディングボックス候補の数
BBOX_COUNT = 2

# african-wildlifeは 普通 4クラス(仮定).  yaml의 ncが正解.
CLASS_COUNT = 4

BATCH = 8  # OOM 発生時 4/2で
EPOCHS = 50

train_ds = make_dataset(ROOT, "train", IMG_SIZE, GRID_SIZE, CLASS_COUNT, batch=BATCH, shuffle=True)
test_ds = make_dataset(ROOT, "test", IMG_SIZE, GRID_SIZE, CLASS_COUNT, batch=BATCH, shuffle=False)
#val_ds   = make_dataset(ROOT, "val",   IMG_SIZE, S, C, batch=BATCH, shuffle=False)

model = build_yolov1(input_shape=(IMG_SIZE, IMG_SIZE, 3), grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT, backbone_trainable=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Keras compile: lossに dict入力をため、 カスタム loss wrapper 使用
@tf.function
def loss_fn(y_true, y_pred):
    return yolo_v1_loss(y_true, y_pred, grid_size=GRID_SIZE, bbox_count=BBOX_COUNT, class_count=CLASS_COUNT)

model.compile(optimizer=optimizer, loss=loss_fn)

# チェックポイント
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/mini_yolo.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[ckpt])
