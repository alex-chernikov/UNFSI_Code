import os.path

import tensorflow as tf
import cv2

import numpy as np
import pandas as pd

from glob import glob
from os.path import basename, isdir
from tqdm import tqdm

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
matplotlib.use(gui_env[0])
from matplotlib import pyplot as plt

import const


def show(dat,reverse=False,label=None):
    if isinstance(dat,list) or isinstance(dat,tuple):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(np.squeeze(dat[0]))
        mask = np.squeeze(dat[1])
        axarr[1].imshow(mask)
    else:
        img=dat
        if not reverse:
            plt.imshow(img)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if label is not None:
        plt.title(label)
    plt.show()


def copy_images_to_cluster_folders(image_paths, clusters, output_dir):
    import os
    import shutil

    unique_clusters = np.unique(clusters)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster in unique_clusters:
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster}')

        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        cluster_indices = np.where(clusters == cluster)[0]

        for idx in cluster_indices:
            image_path = image_paths[idx]
            shutil.copy(image_path, cluster_dir)


def get_closest_to_centers(features, n_clusters):
    from sklearn.cluster import KMeans
    import numpy as np
    from scipy.spatial.distance import cdist

    kmeans = KMeans(n_clusters=n_clusters, random_state=43)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    distances = cdist(features, cluster_centers, 'euclidean')
    closest_indices = np.argmin(distances, axis=0)
    return closest_indices, kmeans.labels_, cluster_centers



def get_features_model():
    from tensorflow.keras.models import load_model, Model
    model_name_path=const.CLS_PATH
    weights_path= model_name_path.replace('.keras','.weights.h5')

    model = load_model(model_name_path)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    feature_extractor = Model(inputs=model.input, outputs=[model.layers[-2].output,model.layers[-1].output])
    return feature_extractor


def extract_features(images,feature_extractor=None):
    for i,image in enumerate(images):
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        images[i] = image
    inputs=tf.stack(images)
    features = feature_extractor.predict(inputs)
    return features


def xywh2xyxy(box):
    x, y, w, h=box
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return [int(x1), int(y1), int(x2), int(y2)]


def upscale_bounding_boxes(image, box, scale=2):
        shY,shX,_ = image.shape
        x, y, w, h = box
        bbox = [x, y, int(w * scale), int(h * scale)]
        if bbox[2]>shX:
            bbox[2]=shX
            bbox[0]=shX//2
        if bbox[3]>shY:
            bbox[3]=shY
            bbox[1]=shY//2

        if bbox[0] + bbox[2]//2 > shX:
            bbox[0] = shX - bbox[2]//2
        if bbox[0] - bbox[2]//2 < 0:
            bbox[0] = bbox[2]//2

        if bbox[1] + bbox[3]//2 > shY:
            bbox[1] = shY - bbox[3]//2
        if bbox[1] - bbox[3]//2 <0:
            bbox[1] = bbox[3]//2
        return bbox


def resize_image(image, max_dimension):
    height, width = image.shape[:2]

    if height > width:
        scaling_factor = max_dimension / height
    else:
        scaling_factor = max_dimension / width

    new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image


'''
    Images in different formats - lansdcape and portrait, needed to restore in order to apply bounding boxes 
    Sharpness filter, box upscaling
'''
def cut_images(df, dirp, dir_save,sharpness_thresh=None,min_allowed_area=None,box_upscale_coef=None):
    from PIL import Image,ImageOps,ExifTags
    import VideoCutter as vc

    info=[]
    gr_unst = df.groupby('file_name')
    for gr_inx, gr in tqdm(gr_unst):
        fname = gr['file_name'].values[0]
        fpath = dirp + fname
        img=Image.open(fpath)
        img=np.array(img)
        if sharpness_thresh is not None:
            imgrs=img
            if max(img.shape)!=1920:
                imgrs=resize_image(img,1920)

            if vc.gradient_magnitude_sharpness(imgrs)<sharpness_thresh:
                continue

        rotated=False
        if img.shape[0]==max(img.shape):
            rotated=True
            img=np.moveaxis(img,0,1)
        shape_main=img.shape[:2]

        for i in range(gr.shape[0]):
            r=gr.iloc[i,:]
            class_inx=int(r['class'])
            if not rotated:
                box=np.array([r['x'],r['y'],r['w'],r['h']]).astype(float)
            else:
                box = np.array([r['y'], r['x'], r['h'], r['w']]).astype(float)
            box=[int(box[0]*shape_main[1]),int(box[1]*shape_main[0]),int(box[2]*shape_main[1]),int(box[3]*shape_main[0])]
            if box_upscale_coef is not None:
                box=upscale_bounding_boxes(img,box,box_upscale_coef)

            cut_area = 100 * box[2] * box[3] / (shape_main[0] * shape_main[1])
            if min_allowed_area is not None:
                if cut_area<min_allowed_area:
                    continue
            info.append((fname,class_inx,cut_area))
            cut_img = img[box[1]-int(box[3]//2):box[1]+int(box[3]//2), box[0]-int(box[2]//2):box[0]+int(box[2]//2)]

            shape=cut_img.shape
            ratio=shape[0]/shape[1]
            ratio=1/ratio if ratio>1 else ratio
            max_side = max(shape)
            new_img=Image.fromarray(cut_img)
            LLM_IMAGE_SIZE=640
            if (shape[0]>LLM_IMAGE_SIZE or shape[1]>LLM_IMAGE_SIZE) or (shape[0]<LLM_IMAGE_SIZE and shape[1]<LLM_IMAGE_SIZE):
                if shape[0]==max_side:
                    new_img = new_img.resize((int(LLM_IMAGE_SIZE * ratio),LLM_IMAGE_SIZE), resample=Image.LANCZOS)
                else:
                    new_img = new_img.resize((LLM_IMAGE_SIZE, int(LLM_IMAGE_SIZE * ratio)), resample=Image.LANCZOS)
            new_img.save(f'{dir_save}cl{class_inx}i{i}_{fname}')


def get_dataframe_from_yolo_folder(dir_labels):
    files=glob(dir_labels+'*.txt')
    file_names=[]
    data=[]
    for fp in files:
        f=open(fp,'r')
        lines=f.readlines()
        file_names.extend([basename(fp).replace('.txt','.jpg')]*len(lines))
        data.extend([x.replace('\n','').split(' ') for x in lines])
    df=pd.DataFrame(data,columns=['class','x','y','w','h']).astype(float)
    df['file_name']=file_names
    return df


def create_features(files,batch_size,save_name=None):
    import tensorflow as tf
    import Classifier as mn

    feats,pred_classes,probs=[],[],[]

    if isdir(files):
        files=glob(files+'/*.jpg')

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(mn.load_and_preprocess_image_MobNetV2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    feature_extractor=get_features_model()
    for fp in dataset:
        preds=feature_extractor.predict(fp.numpy())
        pred_classes.extend(tf.argmin(preds[1],axis=1).numpy())
        probs.extend(tf.reduce_max(preds[1],axis=1).numpy())
        feats.extend(preds[0])
    feats=pd.concat([pd.Series([basename(x) for x in files],name='file_name'),pd.Series(pred_classes,name='pred_class'),pd.Series(probs,name='prob'),pd.DataFrame(feats)],axis=1)
    if save_name is not None:
        feats.to_csv(const.FEATURES_PATH+save_name+'.xz',compression='xz',index=None)
    else:
        return feats


def split_by_class_and_cluster_features(all_files,all_feats,nclusters=5):
    if (isinstance(all_files,str) and isdir(all_files)):
        all_files=glob(all_files+'*.jpg')

    all_feats=np.array(all_feats).astype(float)
    df=pd.DataFrame([(int(basename(x).split(' ')[0][2]),x) for x in all_files],columns=['class','file_name'])
    files_set={}
    for cl in np.unique(df['class']):
        inxs=df[df.iloc[:,0]==cl].index.values
        files=all_files[inxs].values

        if len(files)<=nclusters:
            print('Class', cl,f'has less images than number of clusters: {len(files)} out of {nclusters} required.',
                  '\nTaking all the samples as representatives.', )
            files_set[cl]=files
            continue
        feats=all_feats[inxs]
        closest_indices, clusters, cluster_centers = get_closest_to_centers(feats, nclusters)
        files_set[cl]=files[closest_indices]
    return files_set