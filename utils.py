from os.path import basename
import numpy as np
from datetime import datetime
import re
from PIL import Image, ImageDraw
import image_processor as ip

pattern2 = r'(?:(?:LONG|VIDEO|VIDEO |Photo|VID)[_-]?(\d+)).*?(\d{2}\.\d{2}\.\d{4}|\d{1,2}[A-Za-z]{3}\d{4}|\d{1,2}[A-Za-z]{3}|[A-Za-z]{3,4}\d{1,4})'


def extract_vendor_date(s):
    pattern=pattern2
    return '_'.join(re.search(pattern, s, re.IGNORECASE).groups())


def extract_class_from_file_name(file_path):
    rclass = re.search(r'(GB|DW|WP|HW|DS|CT|FL|VD|CS|WD)', basename(file_path)).group(0)
    return rclass


def normalize_string_dates(date_str):
    match = re.search(r'(\d+_)?(\d+)([A-Za-z]+)(\d{4})?', date_str)
    if match:
        day = match.group(2)
        month = match.group(3)
        return f'{day}{month}'
    return None


def convert_date_to_string(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f%z')
    return date_obj.strftime('%-d%b')


def date_matcher(list1,list2):
    normalized_list1 = [normalize_string_dates(date) for date in list1]
    normalized_list2 = [convert_date_to_string(date) for date in list2]

    matched_indices = []
    unmatched_list1_indices = []
    unmatched_list2_indices = []

    for i, date1 in enumerate(normalized_list1):
        if date1 in normalized_list2:
            matched_indices.append((i, normalized_list2.index(date1)))
        else:
            unmatched_list1_indices.append(i)

    for j, date2 in enumerate(normalized_list2):
        if date2 not in normalized_list1:
            unmatched_list2_indices.append(j)

    matched_indices, (unmatched_list1_indices, unmatched_list2_indices) = matched_indices, (
    unmatched_list1_indices, unmatched_list2_indices)
    return matched_indices,(unmatched_list1_indices,unmatched_list2_indices)



def draw_multiline_text(image_path, text, position=(50, 50), output_path=None,font_size=30):
    image = Image.open(image_path)
    if max(image.size) != 1920:
        prop=image.size[0]/image.size[1]
        image=np.array(image)
        image=ip.resize_image(image,1920)
        if prop<1:
            image=np.swapaxes(image,0,1)
        image=Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    draw.multiline_text(xy=position, text=text, font=None, fill='white',
                        font_size=font_size,stroke_width=2, stroke_fill='black')

    if output_path:
        output_path_im = output_path+ "/txt_" + image_path.split('/')[-1]
        image.save(output_path_im)
    else:
        return image

    return output_path