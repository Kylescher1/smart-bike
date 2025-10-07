import cv2
import numpy as np
from rknnlite.api import RKNNLite
from postprocess import yolov8_post_process  # comes from model zoo example

MODEL_PATH = '/usr/share/rknn_model_zoo/examples/yolov8/model/yolov8.rknn'
LABELS = [
 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
 'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
 'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
 'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
 'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
 'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
 'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
 'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush'
]


def init_model():
    rknn = RKNNLite()
    print("Loading model...")
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        raise RuntimeError('Failed to load RKNN model')

    print("Initializing runtime...")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        raise RuntimeError('Failed to initialize RKNN runtime')
    return rknn

def preprocess(img):
    img = cv2.resize(img, (640, 640))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    rknn = init_model()
    # --- Debug: inspect model I/O expectations ---
    print("\nModel input/output info:")
    try:
        io_info = rknn.list_inputs_outputs()
        for name, info in io_info.items():
            print(name, "->", info)
    except Exception as e:
        print("Could not query list_inputs_outputs:", e)

    try:
        print("Inference time info:", rknn.query_inference_time())
    except Exception as e:
        print("Could not query inference time:", e)
    # ----------------------------------------------
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess and inference
        input_img = preprocess(frame)
        input_data = np.expand_dims(input_img.transpose(2, 0, 1), 0)
        outputs = rknn.inference(inputs=[input_data])
        print("Inference done, raw output shapes:", [o.shape for o in outputs])

        # postprocess
        boxes, classes, scores = yolov8_post_process(outputs)

        h, w = frame.shape[:2]
        scale_x = w / 640.0
        scale_y = h / 640.0
        objects = []

        for box, cls_id, score in zip(boxes, classes, scores):
            if score < 0.5:
                continue
            label = LABELS[int(cls_id)] if int(cls_id) < len(LABELS) else str(cls_id)

            # rescale coordinates to match camera frame
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            objects.append({"label": label, "confidence": float(score), "bbox": [x1, y1, x2, y2]})

        if not objects:
            print("No objects detected")
        else:
            print(objects)

        display = cv2.resize(frame, (800, 600))
        cv2.imshow("YOLOv8 RKNN", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()

