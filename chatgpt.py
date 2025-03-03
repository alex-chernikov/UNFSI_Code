import openai
import base64
import pandas as pd
from io import StringIO
from PIL import Image
import io

import re

from tqdm import tqdm
from glob import glob

import const


openai.api_key = const.OPENAI_KEY

CHOICES='''
list_name,name,label
yes_no,1,yes
yes_no,2,no
ynu,1,yes
ynu,2,no
ynu,3,unclear
cooperation,1,Weak
cooperation,2,Fair
cooperation,3,Good
cooperation,4,Very good
crowd,1,not crowded
crowd,2,somewhat crowded
crowd,3,very crowded
position,1,I am a relative of the owner
position,2,I am running the business
position,3,I rented this stall
position,4,I am a full-time employee
position,5,I am a part-time employee
position,6,other
enu_name,Gopal,Gopal
enu_name,Joydev,Joydev
enu_name,Manikant,Manikant
enu_name,Nandita,Nandita
enu_name,Soumen,Soumen
enu_name,Sunanda,Sunanda
locaaa,Dalhousie,Dalhousie
locaaa,Rashbihari,Rashbihari
locaaa,Sector_V,Sector V'''


QUESTIONS='''
dcode;question;answer;class
q2_1_1_1;Is the hand washing facility located (a) on the ground or (b) on the knee level or higher above the ground, or (c) Unknown?;LETTER;HW
q2_1_1_2;Does this hand washing facility have a lid?;Yes/No/Unknown;HW
q2_1_1_3;With this hand washing facility is there soap available and maximum an arm's length away from the tap?;Yes/No/Unknown;HW
q2_1_1_4;With this hand washing facility - is used water collected into some tank after hand washing?;Yes/No/Unknown;HW

q2_1_2_1;Is the water tank covered by a lid?;Yes/No/Unknown;WP
q2_1_2_2;Is the water tank cracked or does the tank have holes?;Yes/No/Unknown;WP

q2_1_3_1;Is the washing container located (a) on the ground or (b) on the knee level or higher above the ground, or (c) Unknown?;LETTER;DW
q2_1_3_3;Are the dirty plates, pots, cutlery observable and located (a) right on the ground or (b) in a container protected from the ground, or (c) Unknown if not observable?;LETTER;DW
q2_1_3_4;Is the washing container free of cracks and holes? Answer Yes only if it has no cracks or other damages, answer No in all other cases.;Yes/No/Unknown;DW
q2_1_3_5;Is the ground around the washing container clean from debris (rest of food, other waste)?;Yes/No/Unknown;DW

q2_1_4_1;Is the garbage bin made of (a) hard material or (b) plastic bag and alike?;LETTER;GB
q2_1_4_2;Is the garbage bin cracked or does it have holes?;Yes/No/Unknown;GB
q2_1_4_3;Are there visible animals or insects in or around the garbage bin?;Yes/No/Unknown;GB
q2_1_4_4;Does the area around the garbage bin have standing water?;Yes/No/Unknown;GB

q2_1_7;Is the food displayed protected from direct exposure to sun/rain?;Yes/No/Unknown;DS
q2_1_7_1;Is the food display stall positioned under a roof?;Yes/No/Unknown;DS
q2_1_7_2;Is the food on the food display stall out of the sun?;Yes/No/Unknown;DS'''


CONVERSION_TABLE = '''q2_1_1_1:b 1,a 0,c -1;q2_1_1_2:ynu;q2_1_1_3:ynu;q2_1_1_4:ynu;
 q2_1_2_1:ynu;q2_1_2_2:ynu;
 q2_1_3_1:b 1,a 0,c -1;q2_1_3_3:a 0,b 1,c 0;q2_1_3_4:a 1,b 1,c 0;q2_1_3_5:ynu;
 q2_1_4_1:a 1,b 0;q2_1_4_2:a 1,b 1,c 0;q2_1_4_3:ynu;q2_1_4_4:ynu;
 q2_1_7:ynu;q2_1_7_1:ynu;q2_1_7_2:ynu'''


CONVERSION_TABLE=CONVERSION_TABLE.replace('\n','').split(';')


temp=[]
for x in CONVERSION_TABLE:
    cont=x.split(':')
    conv=cont[1].strip()
    if not conv=='ynu':
        conv={k: int(v) for k, v in (item.split() for item in conv.split(','))}
    temp.append((cont[0].strip(),conv))
CONVERSION_TABLE=pd.DataFrame(temp,columns=['dcode','answer'])
YNU_CONVERSION={'yes':1,'no':0,'unknown':0}

reply_pattern = r'[{}[\]()<>,.!?;:"\'\\/-]'


def convert_detailed_to_ynu(reply):
    row=CONVERSION_TABLE[CONVERSION_TABLE['dcode']==reply['dcode']]# ['answer']

    short_reply = re.sub(reply_pattern, '', reply['short_ans']).lower()

    if not short_reply=='unknown' and QUESTIONS_ANSWER_TYPE[reply['dcode']]=='LETTER':
        try:
            if short_reply=='aa':
                short_reply='a'
            ynu_reply=row['answer'].values[0][short_reply]
        except:
            print(reply)
            return
    else:
        if short_reply=='true':
            short_reply='yes'
        if short_reply=='false':
            short_reply='no'
        try:
            ynu_reply=YNU_CONVERSION[short_reply]
        except:
            print('Cannot extract short answer from:',reply['answer'])
            return 0
    return ynu_reply


def extract_short_answer(answer):
    sa=answer.strip().replace('\n',' ').split(' ')[0]
    return re.sub(reply_pattern, '', sa).lower()



def prepare_question(question,answer):
    return f'{question} Answer type- {answer}'

csv_data = StringIO(CHOICES)
CHOICES = pd.read_csv(csv_data)

csv_data = StringIO('\n'.join([x for x in QUESTIONS.split('\n') if len(x)>0]))
QUESTIONS = pd.read_csv(csv_data,sep=';')
QUESTIONS_ANSWER_TYPE={x[0]:x[1] for x in zip(QUESTIONS['dcode'],QUESTIONS['answer'])}


def merge_gtp_answers_with_questions(fname):
    df=pd.read_csv(const.RESULTS_PATH+f'LLM/{fname}.xz',compression="xz")
    df['gpt_reply']=df['short_ans'].str.lower() + ') ' + df['explanation']
    questions=QUESTIONS.copy()
    questions['qnum'] = questions.groupby('class').cumcount() + 1

    merged_df = pd.merge(df, questions[['class','qnum','question','answer','dcode']], on=['class', 'qnum'], how='left')
    return merged_df


def send_image_and_ask_question(file, ncode):
    def encode_image(image_path):
        with Image.open(image_path) as img:
            width, height = img.size

            # Determine the new dimensions
            if width!=640 and height!=640:
                if width > height:
                    new_width = 640
                    new_height = int((640 / width) * height)
                else:
                    new_height = 640
                    new_width = int((640 / height) * width)

                img = img.resize((new_width, new_height), Image.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_base64

    base64_image=encode_image(file)

    questions=QUESTIONS[QUESTIONS['class'] == ncode]
    question='\n'.join([f'({x[0] + 1}) {prepare_question(x[1], x[2])}' for x in
               zip(range(len(questions['question'])), questions['question'].values, questions['answer'].values)])
    question+='\n'+'Answer the questions above and return JSON lines format. The fields should be: [question number,LETTER or Yes/No/Unknown,brief explanation], fields names are [qnum,short_ans,explanation]'

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    answer=response.choices[0].message.content
    return answer

def get_chatGPT_GT(image_folder,class_inx,output_file_name='ChatGPT4o_train_cut',images=None):
    import os
    import base64
    import pickle as pkl
    from os.path import basename

    if images is None:
        representatives = pkl.load(open(const.FEATURES_PATH + 'train_cut_representatives.pkl', 'rb'))
        images = representatives[class_inx][:]

    chatgpt_file = const.RESULTS_PATH + f'LLM/{output_file_name}_{class_inx}.txt'

    filter_out_files = []
    if os.path.exists(chatgpt_file):
        df = pd.read_csv(chatgpt_file, sep='\t')
        filter_out_files = df.iloc[:, 1].to_list()

    with open(chatgpt_file, 'a') as f:
        for im_path_name in tqdm(images):
            im_name=basename(im_path_name)
            if im_name in filter_out_files:
                continue
            try:
                imp = os.path.join(image_folder, im_name)
                answer = send_image_and_ask_question(file=imp, ncode=const.CLASSES[class_inx], yes_no=False)
                f.write(f'{const.CLASSES[class_inx]}\t{im_name}\t{base64.b64encode(answer.encode()).decode()}\n')
            except Exception as e:
                print(e)
                print(im_name,'-------------------------------------')


def combine_chatgpt_descriptions(file_name='ChatGPT4o_train_cut'):
    import os
    import json
    res=[]
    files=glob(const.RESULTS_PATH + f'LLM/{file_name}*.txt')
    for fp in files:
        class_inx=int(os.path.splitext((os.path.basename(fp)))[0][-1])
        class_name=const.CLASSES[class_inx]
        df=pd.read_csv(fp, sep='\t',index_col=None)
        df.columns=['class','file_name','description']
        for fn,x in zip(df['file_name'].values,df['description'].values):
            info=base64.decodebytes(x.encode()).decode()
            info=re.sub(r'[\n\[\]`\t]', '', info.strip().replace('jsonlines\n','').replace('jsonl\n','').replace('json\n',''))
            info=re.sub(r'\s{2,}', '', info)
            info=info.replace('Unknown,','"Unknown",')
            try:
                inf=json.loads(info)
            except:
                try:
                    inf = info.split('\n')
                    inf = inf[1:-1]
                    inf=json.loads(''.join(inf))
                except:
                    json_objects = re.findall(r'{[^{}]*}', info)
                    njson_objects=[]
                    for obj in json_objects:
                        nobj=obj.strip()
                        nobj=('{' if not obj.startswith('{') else '') + nobj
                        nobj=nobj+('}' if not nobj.endswith('}') else '')
                        njson_objects.append(nobj)
                    inf = [json.loads(obj) for obj in njson_objects]
            for x in inf:
                x['file_name']=fn
                x['class']=class_name
            res.extend(inf)
    df=pd.DataFrame(res)
    df['short_ans']=df['short_ans'].replace({'A':'a','B':'b','C':'c','n/a':'Unknown'})
    df.to_csv(const.RESULTS_PATH+f'LLM/{file_name}_output.xz',index=None,compression="xz")