import tensorflow as tf
import numpy as np
from pathlib import Path

## parsing label
def parse_label_file(label_path: str):
    # label_path: bytes -> str
    label_path = label_path.decode("utf-8")
    if not Path(label_path).exists():
        return np.zeros((0, 5), dtype=np.float32)

    lines = Path(label_path).read_text().strip().splitlines()
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        c, x, y, w, h = parts
        rows.append([float(c), float(x), float(y), float(w), float(h)])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

## encoding target
def encode_yolov1_targets(labels_np: np.ndarray, grid_size: int, class_count: int):
    # labels_np: (N,5) [cls, x, y, w, h] normalized to 0..1
    box = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    obj = np.zeros((grid_size, grid_size, 1), dtype=np.float32)
    cls = np.zeros((grid_size, grid_size, class_count), dtype=np.float32)

    # 一つのCELLに複数の 個体が 入ると 一番大きいボックスを採択
    best_area = np.zeros((grid_size, grid_size), dtype=np.float32)

    for c, x, y, w, h in labels_np:
        cx = int(np.clip(x * grid_size, 0, grid_size - 1))
        cy = int(np.clip(y * grid_size, 0, grid_size - 1))
        area = w * h
        if area <= best_area[cy, cx]:
            continue
        best_area[cy, cx] = area

        x_cell = x * grid_size - cx
        y_cell = y * grid_size - cy

        box[cy, cx] = [x_cell, y_cell, w, h]
        obj[cy, cx, 0] = 1.0
        cls[cy, cx, :] = 0.0
        cls[cy, cx, int(c)] = 1.0

    return box, obj, cls

##使えるdatasetを作る
def make_dataset(root_dir: str, split: str, img_size: int, grid_size: int, class_count: int, batch: int, shuffle=True):
    root = Path(root_dir)
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    # イメージ 拡張者は多様な形になる可能性あるので、 globは 2個
    img_paths = sorted([str(p) for p in img_dir.glob("*.jpg")] + [str(p) for p in img_dir.glob("*.png")])

    def img_to_label_path(img_path: str):
        p = Path(img_path)
        return str(lbl_dir / (p.stem + ".txt"))

    lbl_paths = [img_to_label_path(p) for p in img_paths]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, len(img_paths)), reshuffle_each_iteration=True)

    def _load(img_path, lbl_path):
        # 1) イメージロード/前処理
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)  # pngなら decode_pngに 変えてもいい
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0

        # 2) ラベル ロード/エンコード (py_function)
        def _py(lbl_path_bytes):
            labels = parse_label_file(lbl_path_bytes)
            box, obj, cls = encode_yolov1_targets(labels, grid_size=grid_size, C=class_count)
            return box, obj, cls

        box, obj, cls = tf.py_function(_py, inp=[lbl_path], Tout=[tf.float32, tf.float32, tf.float32])
        box.set_shape((grid_size, grid_size, 4))
        obj.set_shape((grid_size, grid_size, 1))
        cls.set_shape((grid_size, grid_size, class_count))

        y_true = {"box": box, "obj": obj, "cls": cls}
        return img, y_true

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds
