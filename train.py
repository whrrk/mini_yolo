import tensorflow as tf
from model import build_yolov1
from loss import yolo_v1_loss
from dataset_tf import make_dataset

ROOT = "data"
IMG_SIZE = 320
S = 7
B = 2

# african-wildlifeは 普通 4クラス(仮定).  yaml의 ncが正解.
C = 4

BATCH = 8  # OOM 発生時 4/2で
EPOCHS = 50

train_ds = make_dataset(ROOT, "train", IMG_SIZE, S, C, batch=BATCH, shuffle=True)
test_ds = make_dataset(ROOT, "test", IMG_SIZE, S, C, batch=BATCH, shuffle=False)
#val_ds   = make_dataset(ROOT, "val",   IMG_SIZE, S, C, batch=BATCH, shuffle=False)

model = build_yolov1(input_shape=(IMG_SIZE, IMG_SIZE, 3), S=S, B=B, C=C, backbone_trainable=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Keras compile: lossに dict入力をため、 カスタム loss wrapper 使用
@tf.function
def loss_fn(y_true, y_pred):
    return yolo_v1_loss(y_true, y_pred, S=S, B=B, C=C)

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
