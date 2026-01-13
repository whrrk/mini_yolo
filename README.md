This project implements a YOLOv1-style object detection model using TensorFlow (Keras).

 ・YOLO-format annotations (class x y w h) are converted into S×S grid-based training targets.

 ・A CNN backbone (MobileNetV2) is used for feature extraction, and a custom YOLO detection head predicts bounding boxes and classes.

 ・The training uses a YOLOv1-style loss with responsible box selection (IoU-based) and separate coord / object / no-object / class losses.

 ・During inference, model outputs are decoded into image coordinates and filtered using Non-Maximum Suppression (NMS).

 ・The system supports training, validation, and inference on custom YOLO-labeled datasets.