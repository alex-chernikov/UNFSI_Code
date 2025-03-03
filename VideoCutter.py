import numpy as np
import pandas as pd
import cv2
from PIL import Image

import const


def filter_dups(images, output_path, prefix):

    def find_close_pairs(arr, threshold=20000):
        import imagehash
        sorted_arr = arr

        close_pairs = []

        for i in range(len(sorted_arr) - 1):
            for j in range(i + 1, len(sorted_arr)):
                im1=Image.fromarray(images[i][1])
                im2 = Image.fromarray(images[j][1])
                hash1 = imagehash.phash(im1)
                hash2 = imagehash.phash(im2)
                if hash1 - hash2 <= threshold:
                    close_pairs.append((images[i][0], images[j][0], hash1 - hash2))
        return close_pairs
    inxs=pd.DataFrame([(x[0],x[2]) for x in images],columns=['frame','shrp'])
    pairs = find_close_pairs(images, threshold=16)
    pairs = pd.DataFrame(pairs, columns=['u1', 'u2', 'score'])
    keep=set(inxs['frame'].values) - set(pairs['u2'])
    inxs_keep=inxs[inxs.frame.isin(keep)].index.values
    for i in inxs_keep:
        cv2.imwrite(output_path+f'/{prefix}_{images[i][0]}_{int(images[i][2])}.jpg',images[i][1])


def cutVideo(input_path,output_path=None,movement_threshold = 2.0,od_acc=0.9):
    import image_processor as ip
    from time import time
    from tqdm import tqdm

    import yolo
    yolo_model=yolo.init_model()
    feature_extractor=ip.get_features_model()
    classes,cuts=[],[]

    cap = cv2.VideoCapture(input_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def calculate_optical_flow(prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)

    ret, prev_frame = cap.read()
    prev_frame=ip.resize_image(prev_frame,800)
    static_frames = []

    t=time()
    i=-1
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, curr_frame_or = cap.read()
            i+=1
            if not ret:
                break

            pbar.update(1)
            pbar.refresh()
            if i % 3 > 0:
                continue

            curr_frame=ip.resize_image(curr_frame_or,800)
            motion = calculate_optical_flow(prev_frame, curr_frame)
            if motion < movement_threshold:
                static_frames.append(curr_frame)

            prev_frame = curr_frame

            confs, labels, boxes=yolo.yolo_predict(yolo_model,[curr_frame],min_conf=od_acc)
            boxes[:,[0,2]]=boxes[:,[0,2]]*im_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * im_height
            boxes=boxes.astype(int)

            boxes=[ip.xywh2xyxy(ip.upscale_bounding_boxes(curr_frame_or,x,1.25)) for x in boxes.tolist()]
            cut_objects=[curr_frame_or[y1:y2, x1:x2, :] for x1, y1, x2, y2 in boxes]
            cuts.extend(cut_objects)
            classes.extend(labels.astype(int))
    cap.release()
    print(time()-t)
    t=time()

    if len(cut_objects)>0:
        feats=ip.extract_features(cuts,feature_extractor=feature_extractor)[0]
    print(time()-t)

    classes=np.array(classes)
    centers={x: ip.get_closest_to_centers(feats[classes == x], n_clusters=5)[0] for x in np.unique(classes)}
    centers={k:np.where(classes==k)[0][v] for k,v in centers.items()}
    reprs={}
    for k,v in centers.items():
        cut_objects=[]
        for i,ii in enumerate(v):
            cut_object=(cuts[ii].numpy()*255).astype(np.uint8)
            impath=f'{output_path}/{const.CLASSES[k]}_static_frame_{i}.jpg'
            cv2.imwrite(impath, cut_object)
            cut_objects.append(impath)
        reprs[const.CLASSES[k]]=cut_objects
    return reprs