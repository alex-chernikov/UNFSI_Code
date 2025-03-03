import os
import sys
import getopt

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import chatgpt as gpt


from glob import glob
from os.path import basename

import const
import image_processor as ip
import utils
import yolo
import Classifier as cls
import paligemma as pg
import florence as fl

import re
pattern2 = r'(?:(?:LONG|VIDEO|VIDEO |Photo|VID)[_-]?(\d+)).*?(\d{2}\.\d{2}\.\d{4}|\d{1,2}[A-Za-z]{3}\d{4}|\d{1,2}[A-Za-z]{3}|[A-Za-z]{3,4}\d{1,4})'


import torch
print(torch.version.cuda)
print(os.environ.get('CUDA_PATH'))

train_im_folder = const.DATA_FOLDER + 'train/images/'
train_labels = const.DATA_FOLDER + 'train/labels/'
val_labels = const.DATA_FOLDER + 'val/labels/'
val_im_folder = const.DATA_FOLDER + 'val/images/'
cut_train_im_folder = const.DATA_FOLDER + 'cutTrain/'
cut_val_im_folder = const.DATA_FOLDER + 'cutVal/'
llm_val_impath = const.DATA_FOLDER + 'llmVal/'
llm_cut_val_impath = const.DATA_FOLDER + 'llmValCut/'


for dirp in [train_im_folder, train_labels, val_labels, val_im_folder, cut_train_im_folder, cut_val_im_folder]:
    os.makedirs(dirp, exist_ok=True)


def extract_vendor_date(s):
    pattern=pattern2
    return '_'.join(re.search(pattern, s, re.IGNORECASE).groups())


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns=sorted_anns[0:3]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def plot_yolo_annotations(image_path, yolo_annotations):
    """
    Plot YOLO annotations on the given image.

    Parameters:
    image_path (str): Path to the image file.
    yolo_annotations (list of lists): YOLO annotations, each list contains
                                      [class, x_center, y_center, width, height].
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = image.shape

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each annotation
    for annotation in yolo_annotations:
        class_id, x_center, y_center, bbox_width, bbox_height = annotation

        # Calculate the coordinates of the bounding box
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height
        x_min = x_center - (bbox_width / 2)
        y_min = y_center - (bbox_height / 2)
        rect = patches.Rectangle((x_min, y_min), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.text(x_min, y_min, str(class_id), color='white', verticalalignment='top', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 1})

    plt.show()


def step_cut_images():
    df = ip.get_dataframe_from_yolo_folder(dir_labels=val_labels)
    ip.cut_images(df, sharpness_thresh=None, dirp=val_im_folder, dir_save=cut_val_im_folder,
                  min_allowed_area=10, box_upscale_coef=1.25)

    df = ip.get_dataframe_from_yolo_folder(dir_labels=train_labels)
    ip.cut_images(df, sharpness_thresh=None, dirp=train_im_folder, dir_save=cut_train_im_folder, min_allowed_area=10,
                  box_upscale_coef=1.25)


def step_create_classifier():
    from os.path import basename

    files = glob(cut_val_im_folder+'/*.jpg')
    info=[(int(re.search(r'cl(\d+)', basename(x)).group(1)),basename(x)) for x in files]
    vdf=pd.DataFrame(info, columns=['class', 'file_name'])
    vdf=vdf.sort_values(by='file_name', ascending=False)
    vdf=vdf[~vdf.index.isin(vdf[vdf['class']==1].index[::2])].reset_index(drop=True)

    files = glob(cut_train_im_folder+'/*.jpg')
    info=[(int(re.search(r'cl(\d+)', basename(x)).group(1)),basename(x)) for x in files]
    sdf=pd.DataFrame(info, columns=['class', 'file_name'])
    sdf=sdf.sort_values(by='file_name', ascending=False)

    fnames = [extract_vendor_date(x) for x in sdf['file_name']]
    sdf['video_id'] = fnames

    df_cl0 = sdf[sdf['class'] == 0]
    inx_cl0=set(df_cl0.index)

    df_cl1 = sdf[sdf['class'] == 1]
    inx_cl1 = set(df_cl1.index[::2])

    df_cl2 = sdf[sdf['class'] == 2]
    inx_cl2 = set(df_cl2.index)

    df_cl3 = sdf[sdf['class'] == 3]
    inx_cl3 = set(df_cl3.index)

    tot_inx = list(inx_cl0) + list(inx_cl1) + list(inx_cl2) + list(inx_cl3)
    sdf = sdf[sdf.index.isin(tot_inx)]

    cls.train_cls('fv52dr',cut_train_im_folder + sdf['file_name'].values, sdf['class'].values,
             cut_val_im_folder + vdf['file_name'].values, vdf['class'].values, batch_size=2048,ep=5000)


def get_survey_image_info():
    fpath1 = const.DATA_FOLDER + 'Videos/Attempt10_R1/**/*.jpg'
    fpath2 = const.DATA_FOLDER + 'Videos/Attempt10_R2/**/*.jpg'
    files = glob(fpath1, recursive=True)
    files.extend(glob(fpath2, recursive=True))
    res = []
    for x in files:
        rbasename = basename(x)
        tmp = extract_vendor_date(rbasename).split('_')
        print(rbasename)
        rclass = utils.extract_class_from_file_name(rbasename)
        res.append((tmp[0], tmp[1], rclass, x))
    res = pd.DataFrame(res, columns=['vendor', 'date', 'class', 'file_path'])
    return res


def evaluate_classifier():
    models=glob(const.MODELS_PATH+'Classification/WIN/*.keras')
    models={x:x.replace('keras','weights.h5') for x in models if os.path.exists(x.replace('keras','weights.h5'))}
    files = glob(cut_val_im_folder+'/*.jpg')
    info=[(int(re.search(r'cl(\d+)', basename(x)).group(1)),x) for x in files]
    vdf=pd.DataFrame(info, columns=['class', 'file_name'])
    vdf=vdf.sort_values(by='file_name', ascending=False)
    vdf=vdf[~vdf.index.isin(vdf[vdf['class']==1].index[::2])].reset_index(drop=True)
    for mod_path,mod_weights in models.items():
        cls.evaluate_cls(mod_path, weights=mod_weights, paths=vdf['file_name'].values, labels=vdf['class'].values, plot=True)
    exit()


'''
    Plots results of several VLLMs, auto adjusts figure to the number of models
'''
def plot_final_results(df,file_name=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    df.replace(3,0,inplace=True)
    df.replace(-1, 0, inplace=True)

    cols=df.columns
    gt=cols[3]
    df[gt] = df[gt].astype(int)
    models=cols[4:].values

    metrics={}
    for model in models:
        df[model]=df[model].astype(int)
        metrics[model]={}
        metrics[model]['Accuracy'] = accuracy_score(df['survey'], df[model])
        metrics[model]['Precision'] = precision_score(df['survey'], df[model])
        metrics[model]['Recall'] = recall_score(df['survey'], df[model])
        metrics[model]['F1-Score'] = f1_score(df['survey'], df[model])

    metrics_df = pd.DataFrame(metrics)
    sns.set(font_scale=1.25)
    fig, ax = plt.subplots(figsize=(len(models)+2,5))
    sns.heatmap(metrics_df, annot=True, cmap='coolwarm', cbar=False, ax=ax, annot_kws={"size": 16})

    ax.set_title(None)
    ax.set_xlabel(None, fontsize=10)
    ax.set_ylabel(None, fontsize=10)

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(const.RESULTS_PATH+f'{file_name}.jpg')
    plt.show()
    print(metrics_df)


def evaluate_human_survey_vs_vllms():
    tdf = gpt.merge_gtp_answers_with_questions('ChatGPT4o_LLM_val_output')
    gpt_replies = []
    for i in range(tdf.shape[0]):
        row = tdf.iloc[i, :]
        vendor, date = extract_vendor_date(row['file_name']).split('_')
        res = (vendor, date, row['file_name'], row['class'], row['dcode'], row['explanation'])
        gpt_replies.append(res + (gpt.convert_detailed_to_ynu(row),))
    chat_gpt_answers = pd.DataFrame(gpt_replies,
                                    columns=['vendor', 'date', 'file_name', 'class', 'question', 'explanation',
                                             'short_answer'])
    chat_gpt_answers['date'] = [utils.normalize_string_dates(x) for x in chat_gpt_answers['date']]
    chat_gpt_answers['vendor'] = chat_gpt_answers['vendor'].astype(str)
    # aux_llms=['paligemma_val.csv','florence2_val.csv','qwen_val.csv']
    aux_llms_names={'Paligemma':'paligemma_val.csv','Florence2':'florence2_val.csv'}
    aux_llms=[pd.read_csv(f'{const.RESULTS_PATH}LLM/{x}') for x in aux_llms_names.values()]
    for i,df in enumerate(aux_llms):
        df['answer']=df['answer'].fillna('no').astype(str)
        df[f'short_ans']=[gpt.extract_short_answer(x) for ii,x in enumerate(df['answer'])]
        df[f'short_ans']=[gpt.convert_detailed_to_ynu(df.iloc[ii,:]) for ii,x in enumerate(df[f'short_ans'])]
        df=df[['file_name','dcode','short_ans','answer']]
        df.columns=[f'file_name_{i}','dcode',f'short_ans_{i}',f'answer_{i}']
        chat_gpt_answers=chat_gpt_answers.merge(df, left_on=['file_name','question'],right_on=[f'file_name_{i}','dcode'], how='left')
        columns = [f'file_name_{i}']
        columns+=[col for col in ['dcode_x','dcode_y'] if col in chat_gpt_answers.columns]
        chat_gpt_answers = chat_gpt_answers.drop(columns=columns)
    for col in chat_gpt_answers.columns:
        if 'short_ans' in col:
            chat_gpt_answers[col] = chat_gpt_answers[col].astype(int)

    '''
        subfunction corrects the survey with expert opinion in cases of disagreements of survey team and ChatGPT4o
    '''
    def corrrect_survey_with_expert_control():
        dfs=pd.read_csv(const.DATA_FOLDER+'expert_control.csv')
        dfs=dfs.melt(id_vars=['Name']).dropna()
        def make_id(x):
            vid=extract_vendor_date(x).split('_')
            vid=vid[0]+'_'+utils.normalize_string_dates(vid[1])
            return vid
        dfs['id']=[make_id(x) for x in dfs['Name']]
        dfn=pd.read_csv(const.DATA_FOLDER+'vllms_results.csv',sep=';')
        dfn['id']=dfn['vendor'].astype(str)+'_'+dfn['date']
        dfn=dfn[dfn['llm']=='survey'].reset_index(drop=True)
        dfs.columns=['file_name','dcode','short_ans','id']
        dfs.loc[dfs['short_ans']==2,'short_ans']=0
        merged=dfn.merge(dfs,how='left',on=['id','dcode'])
        merged['vendor']=merged['vendor'].astype(str)
        # nadf=merged.loc[merged['short_ans_y'].isna(),:]
        nonna=merged.loc[~merged['short_ans_y'].isna(),:]
        nonna['short_ans_x']=nonna['short_ans_x'].astype(int)
        nonna['short_ans_y']=nonna['short_ans_y'].astype(int)
        disag=nonna[nonna['short_ans_x']!=nonna['short_ans_y']]
        print('Total disagreement Expert-Survey team:',len(disag))
        nonna=nonna[['vendor','date','dcode','short_ans_y','short_ans_x']]
        nonna.columns=['vendor','date','dcode_x','short_answer','Human']

        merged['short_ans_x']=merged['short_ans_y']
        merged=merged[['vendor','date','dcode','short_ans_x']]
        merged.columns=['vendor','date','dcode','short_answer']
        merged=merged.dropna().reset_index(drop=True)
        merged = merged.rename(columns={'dcode_x': 'dcode'})
        nonna = nonna.rename(columns={'dcode_x': 'dcode'})
        return merged,nonna


    def get_survey_info():
        questions = gpt.QUESTIONS
        csvs = [const.DATA_FOLDER+'monitoring_Attempt_10_round1.csv',
                const.DATA_FOLDER+'monitoring_Attempt_10_round2.csv']

        def extract_pattern(text):
            pattern = r'q\d+(_\d+)+'
            match = re.search(pattern, text)
            if match:
                return match.group(0)
            else:
                return text

        def get_sub_df(fpath):
            df = pd.read_csv(fpath)
            if 'sur_stattus' in df.columns:
                df = df[df['sur_stattus'] == 'Done']

            df.columns = [extract_pattern(x) for x in df.columns]
            try:
                df = df[['vendor', 'datee'] + questions['dcode'].tolist()]
            except:
                print(fpath)
                df = None
            return df

        df = []
        for x in csvs:
            sdf = get_sub_df(x)
            if not sdf is None:
                df.append(sdf)

        df = pd.concat(df, axis=0)

        def process_survey_answers(df):
            df = df.fillna('-1')
            df[df == 2] = 0
            # df[df == 2] = 'no'
            return df

        df = process_survey_answers(df)
        df['datee'] = [utils.convert_date_to_string(x) for x in df['datee']]
        id_vars = ['vendor', 'datee']
        value_vars = [col for col in df.columns if col.startswith('q')]
        df[value_vars]=df[value_vars].astype(int)
        df['vendor'] = df['vendor'].astype(str)
        meltdf = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='question', value_name='answer')
        meltdf['short_ans_surv'] = meltdf['answer'].astype(int)

        res_df=pd.merge(chat_gpt_answers,meltdf, left_on=['vendor','date','question'],right_on=['vendor','datee','question'], how='left')
        res_df = res_df.rename(columns={'question': 'dcode'})
        return res_df

    res_df=get_survey_info()
    res_df=res_df[~res_df['short_ans_surv'].isna()]

    corrected,disagreement=corrrect_survey_with_expert_control()
    '''
        Filter all the LLMs replies only on disagreement table
    '''
    res_disagr=res_df.merge(disagreement, how='left', on=['vendor', 'date', 'dcode'])
    res_disagr=res_disagr[~res_disagr['short_answer_y'].isna()]
    res_disagr['short_ans_surv']=res_disagr['short_answer_y'].astype(int)
    res_disagr.rename(columns={'short_answer_x': 'short_answer'}, inplace=True)

    res_df=res_df.merge(corrected, how='left', on=['vendor', 'date', 'dcode'])
    res_df.loc[~res_df['short_answer_y'].isna(),'short_ans_surv']=res_df.loc[~res_df['short_answer_y'].isna(),'short_answer_y']
    res_df.rename(columns={'short_answer_x': 'short_answer'}, inplace=True)

    def prepare_df_for_analysis(df):
        col_list=['vendor', 'date', 'dcode', 'short_ans_surv', 'short_answer']+[f'short_ans_{i}' for i in range(len(aux_llms_names))]
        name_list=['vendor','date','dcode','survey','ChatGPT4o']+list(aux_llms_names.keys())
        if 'Human' in list(df.columns):
            col_list.append('Human')
            name_list.append('Human')
        dfh=df[col_list]
        dfh.columns=name_list
        return dfh

    dfr=prepare_df_for_analysis(res_df)
    plot_final_results(dfr,'final_eval_full')
    dfr=prepare_df_for_analysis(res_disagr)
    plot_final_results(dfr,'final_eval_disagr')



if __name__ == '__main__':
    sys_params=sys.argv[1:]
    params=['mode=']
    mode, sat=None, None
    parameters={}
    try:
        opts, args = getopt.getopt(sys_params, '', params)
        parameters = dict([(x[0].replace('--', ''), x[1]) for x in opts])
    except getopt.GetoptError:
        print('Only the following params should be used:',params)
        sys.exit(2)

    if len(parameters)>0:
        mode=parameters['mode']
    else:
        print('No parameters passed!')
        sys.exit(2)

    if mode=='train_yolo':
        # Create YOLOv10 classifier model
        yolo.train_YOLOv10()

    if mode=='train_cls':
        # Create classifier model
        step_create_classifier()
        exit()

    elif mode=='eval_cls':
        # Evaluate classifier
        evaluate_classifier()

    elif mode=='gpt_train_data':
        # Get ChatGPT replies for train images
        for cl in list(const.CLASSES.keys()):
            gpt.get_chatGPT_GT(cut_train_im_folder,cl)

        gpt.combine_chatgpt_descriptions()
        exit()


    elif mode=='train_pg':
        # Train Paligemma
        import paligemma as pg
        tdf=gpt.merge_gtp_answers_with_questions('ChatGPT4o_train_cut_output')
        tdf['file_name']=cut_train_im_folder+tdf['file_name']
        pg.create_food_safety_dataset(train_df=tdf,batch_size=8)
        pg.train_model(300000)
        exit()

    elif mode=='train_fl':
        # Train Florence
        import florence as fl
        tdf=gpt.merge_gtp_answers_with_questions('ChatGPT4o_train_cut_output')
        tdf['file_name']=cut_train_im_folder+tdf['file_name']
        fl.create_food_safety_dataset(train_df=tdf,batch_size=4)
        fl.train_florence(bf16=False)

    elif mode=='gpt_test_data':
        # Get ChatGPT replies for validation images
        for cl,clv in const.CLASSES.items():
            images=[x for x in glob(llm_val_impath+"/*.jpg") if utils.extract_class_from_file_name(x)==clv]
            gpt.get_chatGPT_GT(llm_val_impath,cl,output_file_name='ChatGPT4o_LLM_val',images=images)

        # Combine ChatGPT and create DF
        gpt.combine_chatgpt_descriptions(file_name='ChatGPT4o_LLM_val')

    elif mode=='eval_pg':
        # Predict Paligemma on Validation set
        tdf=gpt.merge_gtp_answers_with_questions('ChatGPT4o_LLM_val_output')
        tdf['file_name']=llm_val_impath+tdf['file_name']
        pg.create_food_safety_dataset(val_df=tdf)
        mod_name='v2_foodSafety_PaliGemma-3b-224px_150000ex_bf16'
        pg.prepare_model(mod_name)
        pg.food_safety_predictions(to_csv=True)

    elif mode=='eval_fl':
        # Evaluate Paligemma on test set
        tdf=gpt.merge_gtp_answers_with_questions('ChatGPT4o_LLM_val_output')
        tdf['file_name']=llm_val_impath+tdf['file_name']
        fl.predict_dataframe(tdf)

    elif mode=='fin_eval':
        # Final evaluation of Human survey VS ChatGPT vs Lightweight VLLMs
        evaluate_human_survey_vs_vllms()
