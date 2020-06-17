
import numpy as np
import os
import cv2

import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sas7bdat import SAS7BDAT
import pandas
from tqdm import tqdm
from shapely.geometry import LineString,Point

seed = 42
np.random.seed(seed)



def read_bf_landmarks(image, fname):
    '''
    This function can be used to read Bonefinder generated files
    and then organize landmark points than can be used in the subsequent analysis
    '''

    with open(fname) as f:
        content = f.read()
    landmarks = np.array(list(map(lambda x: [float(x.split()[0]), float(x.split()[1])], content.split('\n')[3:-2])))

    cols = image.shape[1]

    landmarks_all_r = np.round(landmarks)
    landmarks_all_r[:, 0] = cols - landmarks_all_r[:, 0]
    landmarks_all_l = np.round(landmarks)
    landmarks_all = {'TR': (landmarks_all_r )[37:-1, :],
                     'FR': (landmarks_all_r)[0:37, :],
                     'TL': (landmarks_all_l)[37: -1, :],
                     'FL': (landmarks_all_l)[0:36, :]}
    return landmarks_all


def calculate_m225_m8(img, landmarks_all, side):
    p1 = landmarks_all[f'F{side}'][14, :]
    p2 = landmarks_all[f'F{side}'][22, :]
    L, M1 = rotate_img(img, p1, p2)



    landmarksf = np.dot(np.append(landmarks_all[f'F{side}'], np.ones((landmarks_all[f'F{side}'].shape[0], 1)), axis=-1),
                        M1.T).astype(int)
    landmarkst = np.dot(np.append(landmarks_all[f'T{side}'], np.ones((landmarks_all[f'T{side}'].shape[0], 1)), axis=-1),
                        M1.T).astype(int)


    if landmarkst[10, 0] < landmarksf[12, 0]:
        landmarksf[12, 0] = landmarkst[10, 0]
    if landmarkst[26, 0] > landmarksf[24, 0]:
        landmarksf[24, 0] = landmarkst[26, 0]

    line1 = LineString(landmarksf[12:25])
    line2 = LineString(landmarkst[10:27])


    points = list(line1.coords)
    p8 = points[0][0] + 0.2 * (points[-1][0] - points[0][0])
    p225 = points[0][0] + 0.775 * (points[-1][0] - points[0][0])

    r_ind = img.shape[1] - 1

    line3 = LineString([(p225, 0), (p225, r_ind)])
    p1 = line1.intersection(line3)
    p2 = line2.intersection(line3)

    jsw225 = p2.y - p1.y

    line4 = LineString([(p8, 0), (p8, r_ind)])
    p3 = line1.intersection(line4)
    p4 = line2.intersection(line4)

    jsw8 = p4.y - p3.y

    return(jsw225,jsw8)


def calculate_min_jsw(landmarks_all, side):

    min_jsw_dist = 100

    if side == 'R':
        if landmarks_all['TR'][10,0] < landmarks_all['FR'][12, 0]:
            landmarks_all['FR'][12, 0] = landmarks_all['TR'][10, 0]
        if landmarks_all['TR'][26,0] > landmarks_all['FR'][24, 0]:
            landmarks_all['FR'][24, 0] = landmarks_all['TR'][26, 0]

        line1 = LineString(landmarks_all['FR'][12:25])
        line2 = LineString(landmarks_all['TR'][10:27])
        min_jsw_dist = line1.distance(line2)

    if side == 'L':

        if landmarks_all['TL'][10,0] < landmarks_all['FL'][12, 0]:
            landmarks_all['TL'][10, 0] = landmarks_all['FL'][12, 0]
        if landmarks_all['TL'][26,0] > landmarks_all['FL'][24, 0]:
            landmarks_all['TL'][26, 0] = landmarks_all['FL'][24, 0]
        line1 = LineString(landmarks_all['FL'][12:25])
        line2 = LineString(landmarks_all['TL'][10:27])

        min_jsw_dist = line1.distance(line2)

    return min_jsw_dist


def calculate_jsw_descriptor(landmarks_all, side):
    des = []

    if side =='R':
        for t in landmarks_all['TR'][10:27]:
            for f in landmarks_all['FR'][12:25]:
                dist = np.linalg.norm([(t[0]-f[0]), (t[1]- f[1])])
                des.append(dist)

    if side == 'L':
        for t in landmarks_all['TL'][10:27]:
            for f in landmarks_all['FL'][12:25]:
                dist = np.linalg.norm([(t[0] - f[0]), (t[1] - f[1])])
                des.append(dist)

    return np.array(des)



def rotate_img(img, p1,p2):
    # Based on https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img =  cv2.warpAffine(img, M, (nW, nH), cv2.INTER_CUBIC)

    return img, M


#Based on https://github.com/MIPT-Oulu/KneeOARSIGrading/blob/master/oarsigrading/dataset/metadata/utils.py"
def read_sas7bdat(fpath):
    rows = []
    with SAS7BDAT(fpath) as f:
        for row in f:
            rows.append(row)
    return pandas.DataFrame(rows[1:], columns=rows[0])


def oai_crops():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../DATA/OAI_00m_SIDES_LNDMRKS/')
    parser.add_argument('--assessment', default='../DATA/OAI/kXR_SQ_BU00_SAS/kxr_sq_bu00.sas7bdat')
    parser.add_argument('--save_results', default='../DATA/CROPS/OAI_00m_tm/')


    args = parser.parse_args()
    os.makedirs(args.save_results, exist_ok=True)

    data = pandas.read_sas( args.oai_assessment, format = 'sas7bdat', encoding='iso-8859-1')

    visit = 'V00'
    variables = ['ID',
                 'SIDE',
                 f'{visit}XRKL',
                ]

    data = data[variables]
    data.dropna(inplace=True)
    data.drop_duplicates(subset=['ID', 'SIDE'], inplace= True)
    data = data.rename(columns={'V00XRKL': 'KL'})
    data.KL = data.KL.astype(int)
    data.SIDE = data.SIDE.astype(int)
    kl = [0, 1, 2, 3, 4]
    data = data[data.KL.isin(kl)]
    data.reset_index(drop=True, inplace=True)

    with tqdm(total=len(list(data.iterrows()))) as pbar:
        for index,row in data.iterrows():
            pbar.update(1)
            patientID = row.ID

            side = 'R'
            if int(row.SIDE) == 2: side = 'L'

            f =f'{args.data_root}{patientID}_{side}.npy'
            img,landmarks_all = np.load(f)

            min_jsw = calculate_min_jsw(landmarks_all, side)
            jsw_des = calculate_jsw_descriptor(landmarks_all, side)

            fjsw = calculate_m225_m8(img,landmarks_all,side)

            pr = np.round(landmarks_all[f'T{side}'][26, :])
            pr_lat = np.round(landmarks_all[f'T{side}'][10, :])

            tb_width = abs(pr[0] - pr_lat[0])
            crop_width = int(tb_width / 7)
            if crop_width % 2 == 1:
                crop_width = crop_width + 1

            margin = 0 #experimental, int(crop_width*0.1)
            tmed = img[pr[1] - margin : pr[1] + crop_width, pr[0] - crop_width: pr[0] + margin]
            jsize = float(abs(landmarks_all[f'T{side}'][26, 0] - landmarks_all[f'T{side}'][10, 0]))

            path_save = args.save_results +  str(patientID) + f'_KL_{row.KL}_{side}.npy'
            np.save(path_save, [tmed, min_jsw, jsw_des, fjsw, row.KL, jsize])


def most_crops():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../DATA/MOST_00m_SIDES_LNDMRKS/')
    parser.add_argument('--assessment', default='../DATA/MOST/All_SAS 9.4/mostv01235xray.sas7bdat')
    parser.add_argument('--save_results', default='../DATA/CROPS/MOST_00m_tm/')
    parser.add_argument('--meta_path', default='../DATA/MOST/')

    args = parser.parse_args()
    os.makedirs(args.save_results, exist_ok=True)

    meta_path = args.meta_path
    most_meta = read_sas7bdat(args.assessment)

    #based on https://github.com/MIPT-Oulu/KneeOARSIGrading/oarsigrading/dataset/metadata/most.py
    follow_up_dict_most = {0: '00', 1: '15', 2: '30', 3: '60', 5: '84'}

    most_names_list = pandas.read_csv(os.path.join(meta_path, 'names.txt'), header=None)[0].values.tolist()
    xray_types = pandas.DataFrame(
        list(map(lambda x: (x.split('/')[0][:-5], follow_up_dict_most[int(x.split('/')[1][1])], x.split('/')[-2]),
                 most_names_list)), columns=['ID', 'VISIT', 'TYPE'])

    most_meta_all = []
    for visit_id in [0]:
        for leg in ['L', 'R']:
            features = ['MOSTID', ]
            for compartment in ['L', 'M']:
                for bone in ['F', 'T']:
                    features.append(f"V{visit_id}X{leg}OS{bone}{compartment}"),
                features.append(f"V{visit_id}X{leg}JS{compartment}")
            features.append(f"V{visit_id}X{leg}KL")
            tmp = most_meta.copy()[features]
            trunc_feature_names = list(map(lambda x: 'XR' + x[4:], features[1:]))
            tmp[trunc_feature_names] = tmp[features[1:]]
            tmp.drop(features[1:], axis=1, inplace=True)
            tmp['SIDE'] = int(1 if leg == 'R' else 2)
            tmp = tmp[~tmp.isnull().any(1)]
            tmp['ID'] = tmp['MOSTID'].copy()
            tmp.drop('MOSTID', axis=1, inplace=True)
            most_meta_all.append(tmp)

    most_meta = pandas.concat(most_meta_all)
    most_meta = most_meta[(most_meta[trunc_feature_names] <= 4).all(1)]
    most_meta = pandas.merge(xray_types, most_meta)
    most_meta = most_meta[most_meta.TYPE == 'PA10']
    data = most_meta.drop('TYPE', axis=1)
    data = data[data.VISIT == '00']

    variables = ['ID', 'SIDE', 'XRKL']

    data = data[variables]
    data.dropna(inplace=True)
    data.drop_duplicates(subset=['ID', 'SIDE'], inplace= True)
    data = data.rename(columns={'XRKL': 'KL'})
    data.KL = data.KL.astype(int)
    data.SIDE = data.SIDE.astype(int)
    kl = [0, 1, 2, 3, 4]
    data = data[data.KL.isin(kl)]

    data.reset_index(drop=True, inplace=True)

    with tqdm(total=len(list(data.iterrows()))) as pbar:
        for index, row in data.iterrows():
            pbar.update(1)
            patientID = row.ID

            side = 'R'
            if int(row.SIDE) == 2: side = 'L'

            f = f'{args.data_root}{patientID}_{side}.npy'
            img, landmarks_all = np.load(f)

            min_jsw = calculate_min_jsw(landmarks_all, side)
            jsw_des = calculate_jsw_descriptor(landmarks_all, side)

            fjsw = calculate_m225_m8(img, landmarks_all, side)

            pr = np.round(landmarks_all[f'T{side}'][26, :])
            pr_lat = np.round(landmarks_all[f'T{side}'][10, :])

            tb_width = abs(pr[0] - pr_lat[0])
            crop_width = int(tb_width / 7)
            if crop_width % 2 == 1:
                crop_width = crop_width + 1

            margin = 0  # experimental, int(crop_width*0.1)
            tmed = img[pr[1] - margin: pr[1] + crop_width, pr[0] - crop_width: pr[0] + margin]
            jsize = float(abs(landmarks_all[f'T{side}'][26, 0] - landmarks_all[f'T{side}'][10, 0]))

            path_save = args.save_results + str(patientID) + f'_KL_{row.KL}_{side}.npy'
            np.save(path_save, [tmed, min_jsw, jsw_des, fjsw, row.KL, jsize])


def main():

    dataset = 'OAI' # or MOST
    if dataset == 'OAI':
        oai_crops()
    elif dataset == 'MOST':
        most_crops()
    else:
        print ('Select OAI or MOST Dataset ')


if __name__ == "__main__":
    main()
