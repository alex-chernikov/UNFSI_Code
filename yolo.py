import numpy as np
from ultralytics import YOLOv10
import const


def init_model():
    return YOLOv10(const.YOLO_MODEL_PATH)


def train_YOLOv10():
    cfg = {'data': 'food_un.yaml', 'epochs': 400, 'batch': 80, 'imgsz': 800, 'device': [0], 'mixup':0.5, 'mosaic':0.5, 'shear':0.25, 'cls':6.0, 'box':1.0, 'dfl':3.0,'optimizer':'SGD'}
    model = YOLOv10(model="yolov10n.pt")
    model.train(**cfg)

def yolo_predict(model=None,images=[],batch=256,min_conf=0.8):
    if isinstance(model,str) or model is None:
        model=init_model()

    tconfs, tlabels,tboxes=[],[],[]

    results = model.predict(images, stream=False,batch=batch,verbose=False)
    for result in results:

        bboxes = result.boxes.cpu()
        confs=bboxes.conf.numpy()
        labels=bboxes.cls.numpy()
        boxes=bboxes.xywhn.numpy()

        inxs=np.where(confs>min_conf)
        if len(images)==1:
            return confs[inxs], labels[inxs], boxes[inxs]
        else:
            tconfs.append(confs),tlabels.append(labels),tboxes.append(boxes)
    return tconfs,tlabels,tboxes