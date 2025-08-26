#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luyin
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import os

# Function Definitions

def sort_cnt(filename):
    n_subject = filename.split('_')[1][1:]
    n_cnt = (filename.split('_')[3]).split('.')[0]
    return int(n_subject), int(n_cnt)

def load_cnt(cnt_all, n_subj, name, pth_np, filename):
    cnt = np.load(os.path.join(pth_np, filename))
    if name not in cnt_all[n_subj]:
        cnt_all[n_subj][name] = {'cnt_points': [cnt]}
    else:
        cnt_all[n_subj][name]['cnt_points'].append(cnt)

def load_df(pth_img, pth_np, verbose=False):
    cnt_sutures = [file for file in os.listdir(pth_np) if file.endswith('.npy') and file.split('_')[2] == 'Suture']
    cnt_sutures = sorted(cnt_sutures, key=sort_cnt)

    
    cnt_vessels = [file for file in os.listdir(pth_np) if file.endswith('.npy') and file.split('_')[2] == 'Vessel']
    cnt_vessels = sorted(cnt_vessels, key=sort_cnt)
    
    cnt_all = {}
    
    for filename in cnt_sutures:
        n_subj = filename.split('_')[0] + filename.split('_')[1]
        
        if n_subj not in cnt_all:
            cnt_all[n_subj] = {}
    
    for filename in cnt_vessels:
        n_subj = filename.split('_')[0] + filename.split('_')[1]
        load_cnt(cnt_all, n_subj, 'cnt_vessels', pth_np, filename)
    
    for filename in cnt_sutures:
        n_subj = filename.split('_')[0] + filename.split('_')[1]
        load_cnt(cnt_all, n_subj, 'cnt_sutures', pth_np, filename)
    
    for n_subj in cnt_all.keys():
        if verbose:
            print(f'loading {n_subj}')
        for file in os.listdir(pth_img):
            if file.split('_')[0] + file.split('_')[1] == n_subj:
                img = cv2.imread(os.path.join(pth_img, file))
                cnt_all[n_subj]['img'] = img
    
    return cnt_all

def find_contour_centers(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            centers.append((0, 0))
    return centers

def remove_outliers_using_dbscan(contours, eps=30, min_samples=3):
    centers = find_contour_centers(contours)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    cluster_sizes = np.bincount(labels[labels >= 0])
    largest_cluster_label = np.argmax(cluster_sizes)
    filtered_contours = [contours[i] for i in range(len(contours)) if labels[i] == largest_cluster_label]
    return filtered_contours

def rm_cnt_ol_db(cnt_all):
    for n_subj in cnt_all.keys():
        cnt_sutures = cnt_all[n_subj]['cnt_sutures']['cnt_points']
        img = cnt_all[n_subj]['img']
        cnt_all[n_subj]['cnt_sutures']['cnt_points'] = remove_outliers_using_dbscan(cnt_sutures, eps=int(img.shape[0] * 0.2), min_samples=3)
    return cnt_all

def get_bbx(cnt_all, name):
    for n_subj in cnt_all.keys():
        cnt_sutures = cnt_all[n_subj][name]['cnt_points']
        center_list = []
        box_list = []
        for cnt in cnt_sutures:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center = (int(rect[0][0]), int(rect[0][1]))
            center_list.append(center)
            box_list.append(box)
        cnt_all[n_subj][name]['centers'] = center_list
        cnt_all[n_subj][name]['boxes'] = box_list
    return cnt_all

def draw_dashed_line(img, point1, point2, color, thickness, dash_length=20):
    point1 = np.array(point1)
    point2 = np.array(point2)
    line_length = np.linalg.norm(point2 - point1)
    num_dashes = int(line_length // dash_length)

    for i in range(num_dashes):
        start = point1 + (point2 - point1) * (i / num_dashes)
        end = point1 + (point2 - point1) * ((i + 0.5) / num_dashes)
        start = tuple(start.astype(int))
        end = tuple(end.astype(int))
        cv2.line(img, start, end, color, thickness)

def suture_pca(cnt_all, output_dir, vis=False, suture_pca_calculated=False):
    os.makedirs(output_dir, exist_ok=True)
    for subject, data in cnt_all.items():
        img = data['img'].copy()
        centers = np.array(data['cnt_sutures']['centers'])
        extension_length = 100

        if centers.shape[0] > 1:
            pca = PCA(n_components=2)
            pca.fit(centers)
            center_mean = np.mean(centers, axis=0)
            principal_axis = pca.components_[0]

            projected_centers = np.dot(centers - center_mean, principal_axis)
            min_proj = projected_centers.min()
            max_proj = projected_centers.max()

            point1 = center_mean + (min_proj - extension_length) * principal_axis
            point2 = center_mean + (max_proj + extension_length) * principal_axis

            point1 = tuple(np.round(point1).astype(int))
            point2 = tuple(np.round(point2).astype(int))

            distances_to_axis = np.abs(np.cross(centers - center_mean, principal_axis))

            cnt_all[subject]['cnt_sutures']['pca_axis'] = principal_axis
            cnt_all[subject]['cnt_sutures']['distances_to_axis'] = distances_to_axis

            if suture_pca_calculated:
                normalize_distance = data['cnt_vessels']['aligned_lengths']
                normalized_distances = distances_to_axis / normalize_distance
                cnt_all[subject]['cnt_sutures']['normalized_distances'] = normalized_distances

            if vis:
                draw_dashed_line(img, point1, point2, (255, 0, 0), 3, dash_length=20)

        if vis:
            boxes = data['cnt_sutures']['boxes']
            for center, box in zip(centers, boxes):
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.circle(img, tuple(center), 1, (0, 0, 255), -1)
            cnt_all[subject]['img_bbx'] = img

            output_path = os.path.join(output_dir, f"{subject}.jpg")
            cv2.imwrite(output_path, img)

    return cnt_all

def load_ALI(pth='/home/luyin/Downloads/ALI-Scores-Cluj-Dec23 (1).csv'):
    df = pd.read_csv(pth)
    column_names = [f'Mistake {i}' for i in range(1, 11)]
    missing_columns = [col for col in column_names if col not in df.columns]
    if not missing_columns:
        df['Total Mistakes'] = df[column_names].sum(axis=1)
        df['Count of Mistakes'] = df[column_names].apply(lambda x: (x != 0).sum(), axis=1)
    return df

def crop_rotated_bbx(img, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    x, y, w, h = cv2.boundingRect(box)

    height, width = img.shape[:2]
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = min(w, width - x)
    h = min(h, height - y)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

    result = cv2.bitwise_and(img, img, mask=mask)

    cropped = result[y:y + h, x:x + w]

    return cropped

def reshape_img(image, desired_height=1280):
    aspect_ratio = desired_height / image.shape[0]
    desired_width = int(image.shape[1] * aspect_ratio)
    resized_image = cv2.resize(image, (desired_width, desired_height))
    return resized_image

def vis_ALI(cnt_all, df, n_mistake, output_dir, criterial_dict):
    os.makedirs(output_dir, exist_ok=True)
    ALI_Mistakes = {'Mistake 1': '(Line Disruption)'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img_visall'].copy()

        img = reshape_img(img)
        position = (10, 50)
        font_scale = 1.50
        thickness = 2
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 0)
        e_txt = f'ALI {n_mistake} ' + ALI_Mistakes[f'Mistake {n_mistake}']
        cv2.putText(img, f'{e_txt}: {ALI_score}', position, font_type, font_scale, color, thickness, cv2.LINE_AA)
        if n_mistake == 1:
            position = (position[0], position[1] + 50)
            criteria = criterial_dict[subj_name]
            cv2.putText(img, f'Angular Deviation Max: {criteria:.2f}', position, font_type, font_scale, color, thickness, cv2.LINE_AA)

        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img)

def vis_distribution(cnt_all, name, title, xlabel, ylabel, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    all_list = []
    for subj_name, data in cnt_all.items():
        tgt = data['cnt_sutures'][name]
        plt.scatter([int(subj_name[1:])] * len(tgt), tgt)
        all_list.extend(data['cnt_sutures'][name])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    plt.savefig(os.path.join(save_folder, title) + '.png')
    return all_list

def get_side_lengths(box):
    lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    return lengths

def angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def find_aligned_side(principal_axis, box):
    lengths = get_side_lengths(box)
    vectors = [box[(i + 1) % 4] - box[i] for i in range(4)]
    angles = [angle_between_vectors(principal_axis, vector) for vector in vectors]
    min_angle_index = np.argmin([min(angle, 180 - angle) for angle in angles])
    aligned_length = lengths[min_angle_index]
    return aligned_length

def get_vessel_length(cnt_all):
    for subject, data in cnt_all.items():
        centers = np.array(data['cnt_sutures']['centers'])

        if centers.shape[0] > 1:
            principal_axis = data['cnt_sutures']['pca_axis']
            aligned_lengths = []

            for cnt_vessel in data['cnt_vessels']['cnt_points']:
                box = data['cnt_vessels']['boxes'][0]
                aligned_length = find_aligned_side(principal_axis, box)
                aligned_lengths.append(aligned_length)

            cnt_all[subject]['cnt_vessels']['aligned_lengths'] = aligned_lengths
    return cnt_all

def modify_integer(n):
    return 1 if n > 0 else n

def vis_distribution_with_label(cnt_all, df, n_mistake, name, title, xlabel, ylabel, save_folder):
    ALI_sign_list = []
    lable2color = {0: 'green', 1: 'red'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        ALI_sign_list.append(ALI_sign)
    plt.figure(figsize=(10, 6))
    all_list = []
    legend_True_flag, legend_False_flag = True, True
    for subj_name, data in cnt_all.items():
        tgt = data['cnt_sutures'][name]
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        if legend_True_flag and ALI_sign:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_True_flag = False
        elif legend_False_flag and not ALI_sign:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_False_flag = False
        else:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign])
        all_list.extend(data['cnt_sutures'][name])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_folder, title) + '.png')
    
def vis_distribution_with_label(cnt_all, df, n_mistake, criteria_dict, title, xlabel, ylabel, save_folder):
    ALI_sign_list = []
    lable2color = {0: 'green', 1: 'red'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        ALI_sign_list.append(ALI_sign)
    plt.figure(figsize=(10, 6))
    #all_list = []
    legend_True_flag, legend_False_flag = True, True
    for subj_name, data in cnt_all.items():
        tgt = criteria_dict[subj_name]
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        if legend_True_flag and ALI_sign:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_True_flag = False
        elif legend_False_flag and not ALI_sign:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_False_flag = False
        else:
            plt.scatter([int(subj_name[1:])] * len(tgt), tgt, c=lable2color[ALI_sign])
        #all_list.extend(data['cnt_sutures'][name])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_folder, title) + '.png')

def vis_metric_with_label(cnt_all, df, n_mistake, criteria_dict, title, xlabel, ylabel, save_folder):
    ALI_sign_list = []
    lable2color = {0: 'green', 1: 'red'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        ALI_sign_list.append(ALI_sign)
    plt.figure(figsize=(10, 6))
    #all_list = []
    legend_True_flag, legend_False_flag = True, True
    for subj_name, data in cnt_all.items():
        tgt = criteria_dict[subj_name]
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        if legend_True_flag and ALI_sign:
            plt.scatter([int(subj_name[1:])] , tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_True_flag = False
        elif legend_False_flag and not ALI_sign:
            plt.scatter([int(subj_name[1:])] , tgt, c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_False_flag = False
        else:
            plt.scatter([int(subj_name[1:])], tgt, c=lable2color[ALI_sign])
        #all_list.extend(data['cnt_sutures'][name])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_folder, title) + '.png')

def get_mean_var_max_sum(cnt_all, name):
    mean_list, var_list, max_list, sum_list = [], [], [], []
    for subj_name, data in cnt_all.items():
        tgt = data['cnt_sutures'][name]
        mean_list.append(np.mean(tgt))
        var_list.append(np.std(tgt))
        max_list.append(np.max(tgt))
        sum_list.append(np.sum(tgt))

    return mean_list, var_list, max_list, sum_list

def vis_attr_with_label(cnt_all, arr_list, df, n_mistake, title, xlabel, ylabel, save_folder):
    ALI_sign_list = []
    lable2color = {0: 'green', 1: 'red'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        ALI_sign_list.append(ALI_sign)
    plt.figure(figsize=(10, 6))
    legend_True_flag, legend_False_flag = True, True
    arr_count = 0
    for subj_name, data in cnt_all.items():
        tgt = arr_list
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        if legend_True_flag and ALI_sign:
            plt.scatter([int(subj_name[1:])], tgt[arr_count], c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            arr_count += 1
            legend_True_flag = False
        elif legend_False_flag and not ALI_sign:
            plt.scatter([int(subj_name[1:])], tgt[arr_count], c=lable2color[ALI_sign], label=f'ALI Error = {ALI_sign}')
            legend_False_flag = False
            arr_count += 1
        else:
            plt.scatter([int(subj_name[1:])], tgt[arr_count], c=lable2color[ALI_sign])
            arr_count += 1

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_folder, title) + '.png')

def vis_3d_plot(cnt_all, df, n_mistake, x, y, z):
    ALI_sign_list = []
    subj_num_list = []
    lable2color = {0: 'green', 1: 'red'}
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        ALI_sign = modify_integer(ALI_score)
        ALI_sign_list.append(ALI_sign)
        subj_num_list.append(int(subj_name[1:]))

    labels = ALI_sign_list
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+text',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Picnic'
        ),
        text=subj_num_list,
        textposition='top center'
    )

    layout = go.Layout(
        title='3D Scatter Plot with Plotly',
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    pyo.plot(fig, filename='3d_scatter_plot.html')

def calculate_principal_axis(centers):
    pca = PCA(n_components=2)
    pca.fit(centers)
    principal_axis = pca.components_[0]
    return principal_axis

def compute_short_axis(box, rect, principal_axis, aspect_ratio_threshold=1.2, fixed_length=50):
    lengths = get_side_lengths(box)
    long_side_index = np.argmax(lengths)
    #short_side_index = (long_side_index + 2) % 4
    short_side_index = np.argmin(lengths)
    long_length = lengths[long_side_index]
    short_length = lengths[short_side_index]

    center = np.array(rect[0])
    #print(long_length / short_length)
    if long_length / short_length <= aspect_ratio_threshold:
        side_vectors = [
            box[(i + 1) % 4] - box[i] for i in range(4)
        ]
        side_angles = [angle_between_vectors(side_vector, principal_axis) for side_vector in side_vectors]
        best_side_index = np.argmin([min(angle, 180 - angle) for angle in side_angles])

        aligned_vector = side_vectors[best_side_index]
        unit_aligned_vector = aligned_vector / np.linalg.norm(aligned_vector)
        scaled_aligned_vector = unit_aligned_vector * fixed_length
        short_axis_start = center - scaled_aligned_vector / 2
        short_axis_end = center + scaled_aligned_vector / 2
        short_axis_vector = scaled_aligned_vector
    else:
        next_index = (short_side_index + 1) % 4
        short_axis_vector = box[next_index] - box[short_side_index]
        unit_short_axis_vector = short_axis_vector / np.linalg.norm(short_axis_vector)
        scaled_short_axis_vector = unit_short_axis_vector * fixed_length
        short_axis_start = center - scaled_short_axis_vector / 2
        short_axis_end = center + scaled_short_axis_vector / 2
        short_axis_vector = scaled_short_axis_vector

    return short_axis_start, short_axis_end, short_axis_vector

def get_rotated_bounding_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box, rect

# def order_centers_by_projection(centers, principal_axis):
#     projections = np.dot(centers, principal_axis)
#     sorted_indices = np.argsort(projections)
#     sorted_centers = centers[sorted_indices]
#     return sorted_centers

def order_centers_by_projection(centers, principal_axis):
    # Compute projections of centers along the principal axis
    projections = np.dot(centers - np.mean(centers, axis=0), principal_axis)
    
    if abs(principal_axis[0]) > abs(principal_axis[1]):  # Horizontal principal axis
        if principal_axis[0] > 0:  # Positive slope for horizontal direction
            sorted_indices = np.argsort(projections)
        else:  # Negative slope for horizontal direction
            sorted_indices = np.argsort(projections)[::-1]
    else:  # Vertical principal axis
        if principal_axis[1] > 0:  # Top to bottom
            sorted_indices = np.argsort(projections)
        else:  # Bottom to top
            sorted_indices = np.argsort(projections)[::-1]
    
    # Sort centers based on the projections
    sorted_centers = centers[sorted_indices]
    return sorted_centers



def compute_and_visualize_distances(img, centers):
    num_centers = len(centers)
    distance_list = []
    angle_list = []
    for i in range(num_centers - 1):
        center1 = tuple(np.round(centers[i]).astype(int))
        center2 = tuple(np.round(centers[i + 1]).astype(int))
        distance = np.linalg.norm(np.array(center1) - np.array(center2))

        mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        draw_dashed_line(img, center1, center2, (255, 0, 153), 2, dash_length=20)
        distance_list.append(distance)

    for i in range(num_centers - 2):
        vector1 = centers[i + 1] - centers[i]
        vector2 = centers[i + 2] - centers[i + 1]
        angle = angle_between_vectors(vector1, vector2)
        angle_list.append(angle)
    return distance_list, angle_list




def vis_all(cnt_all, output_dir, vis=True):
    os.makedirs(output_dir, exist_ok=True)

    for subject, data in cnt_all.items():
        
        img = data['img'].copy()
        centers = data['cnt_sutures']['centers']
        if len(centers) > 1:
            centers = data['cnt_sutures']['centers']
            boxes = data['cnt_sutures']['boxes']
            for center, box in zip(centers, boxes):
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            centers = np.array(data['cnt_sutures']['centers'])
            principal_axis = calculate_principal_axis(centers)
            angles_with_principal_axis = []
            
            for cnt_suture in data['cnt_sutures']['cnt_points']:
                box, rect = get_rotated_bounding_box(cnt_suture)
                short_axis_start, short_axis_end, short_axis_vector = compute_short_axis(box, rect, principal_axis, aspect_ratio_threshold=1.2, fixed_length=50)
                angle = angle_between_vectors(short_axis_vector, principal_axis)
                angles_with_principal_axis.append(min(angle, 180 - angle))

                if vis:
                    cv2.line(img, tuple(short_axis_start.astype(int)), tuple(short_axis_end.astype(int)), color=(0, 0, 255), thickness=3)
                    continue
            cnt_all[subject]['cnt_sutures']['angles_with_principal_axis'] = angles_with_principal_axis

            if vis:
                projected_centers = np.dot(centers - np.mean(centers, axis=0), principal_axis)
                min_proj = projected_centers.min()
                max_proj = projected_centers.max()
                center_mean = np.mean(centers, axis=0)
                extension_length = 100
                point1 = center_mean + (min_proj - extension_length) * principal_axis
                point2 = center_mean + (max_proj + extension_length) * principal_axis
                draw_dashed_line(img, point1, point2, color=(255, 0, 0), thickness=3, dash_length=20)

                sorted_centers = order_centers_by_projection(centers, principal_axis)
                distance_list, angle_list = compute_and_visualize_distances(img, sorted_centers)
                cnt_all[subject]['cnt_sutures']['ordered_center'] = sorted_centers
                cnt_all[subject]['cnt_sutures']['center_distance'] = distance_list
                cnt_all[subject]['cnt_sutures']['curvature_angles'] = angle_list

                output_path = os.path.join(output_dir, f"{subject}.jpg")

                centers = data['cnt_vessels']['centers']
                boxes = data['cnt_vessels']['boxes']
                for center, box in zip(centers, boxes):
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
                cnt_all[subject]['img_visall'] = img
                cv2.imwrite(output_path, img)
                print(output_path)

    return cnt_all

def vis_clean(cnt_all, output_dir, vis=True):
    os.makedirs(output_dir, exist_ok=True)

    def add_numbered_bbx(img, centers, boxes, principal_axis):
        for idx, (center, box) in enumerate(zip(centers, boxes)):
            if abs(principal_axis[0]) > abs(principal_axis[1]):  # Horizontal principal axis
                text_position = (center[0], center[1] - 10)
            else:  # Vertical principal axis
                text_position = (center[0] - 10, center[1])
            
            cv2.putText(img, str(idx + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    for subject, data in cnt_all.items():
        img = data['img'].copy()
        centers = data['cnt_sutures']['centers']
        if len(centers) > 1:
            centers = data['cnt_sutures']['centers']
            boxes = data['cnt_sutures']['boxes']
            for center, box in zip(centers, boxes):
                cv2.drawContours(img, [box], 0, (0, 255, 0), 1)

            centers = np.array(data['cnt_sutures']['centers'])
            principal_axis = calculate_principal_axis(centers)
            angles_with_principal_axis = []
            
            for cnt_suture in data['cnt_sutures']['cnt_points']:
                box, rect = get_rotated_bounding_box(cnt_suture)
                short_axis_start, short_axis_end, short_axis_vector = compute_short_axis(box, rect, principal_axis, aspect_ratio_threshold=1.5, fixed_length=50)
                angle = angle_between_vectors(short_axis_vector, principal_axis)
                angles_with_principal_axis.append(min(angle, 180 - angle))

                if vis:
                    #cv2.line(img, tuple(short_axis_start.astype(int)), tuple(short_axis_end.astype(int)), color=(0, 0, 255), thickness=2)
                    continue
            cnt_all[subject]['cnt_sutures']['angles_with_principal_axis'] = angles_with_principal_axis

            if vis:
                projected_centers = np.dot(centers - np.mean(centers, axis=0), principal_axis)
                min_proj = projected_centers.min()
                max_proj = projected_centers.max()
                center_mean = np.mean(centers, axis=0)
                extension_length = 100
                point1 = center_mean + (min_proj - extension_length) * principal_axis
                point2 = center_mean + (max_proj + extension_length) * principal_axis
                #draw_dashed_line(img, point1, point2, color=(255, 0, 0), thickness=2, dash_length=20)

                sorted_centers = order_centers_by_projection(centers, principal_axis)
                # distance_list, angle_list = compute_and_visualize_distances(img, sorted_centers)
                # cnt_all[subject]['cnt_sutures']['ordered_center'] = sorted_centers
                # cnt_all[subject]['cnt_sutures']['center_distance'] = distance_list
                # cnt_all[subject]['cnt_sutures']['curvature_angles'] = angle_list

                add_numbered_bbx(img, sorted_centers, boxes, principal_axis)

                output_path = os.path.join(output_dir, f"{subject}.jpg")

                centers = data['cnt_vessels']['centers']
                boxes = data['cnt_vessels']['boxes']
                for center, box in zip(centers, boxes):
                    cv2.drawContours(img, [box], 0, (0, 0, 0), 2)

                img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
                cnt_all[subject]['img_visall'] = img
                cv2.imwrite(output_path, img)

    return cnt_all

def vis_clean_tempt(cnt_all, output_dir, vis=True):
    os.makedirs(output_dir, exist_ok=True)

    def add_numbered_bbx(img, centers, boxes, principal_axis):
        for idx, (center, box) in enumerate(zip(centers, boxes)):
            if abs(principal_axis[0]) > abs(principal_axis[1]):  # Horizontal principal axis
                text_position = (center[0], center[1] - 30)
            else:  # Vertical principal axis
                text_position = (center[0] - 10, center[1])
            
            #cv2.putText(img, str(idx + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    for subject, data in cnt_all.items():
        img = data['img'].copy()
        centers = data['cnt_sutures']['centers']
        if len(centers) > 1:
            centers = data['cnt_sutures']['centers']
            boxes = data['cnt_sutures']['boxes']
            for center, box in zip(centers, boxes):
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            centers = np.array(data['cnt_sutures']['centers'])
            principal_axis = calculate_principal_axis(centers)
            angles_with_principal_axis = []
            
            for cnt_suture in data['cnt_sutures']['cnt_points']:
                box, rect = get_rotated_bounding_box(cnt_suture)
                short_axis_start, short_axis_end, short_axis_vector = compute_short_axis(box, rect, principal_axis, aspect_ratio_threshold=1.5, fixed_length=50)
                angle = angle_between_vectors(short_axis_vector, principal_axis)
                angles_with_principal_axis.append(min(angle, 180 - angle))

                if vis:
                    #cv2.line(img, tuple(short_axis_start.astype(int)), tuple(short_axis_end.astype(int)), color=(0, 0, 255), thickness=2)
                    continue
            cnt_all[subject]['cnt_sutures']['angles_with_principal_axis'] = angles_with_principal_axis

            if vis:
                projected_centers = np.dot(centers - np.mean(centers, axis=0), principal_axis)
                min_proj = projected_centers.min()
                max_proj = projected_centers.max()
                center_mean = np.mean(centers, axis=0)
                extension_length = 100
                point1 = center_mean + (min_proj - extension_length) * principal_axis
                point2 = center_mean + (max_proj + extension_length) * principal_axis
                #draw_dashed_line(img, point1, point2, color=(255, 0, 0), thickness=2, dash_length=20)

                sorted_centers = order_centers_by_projection(centers, principal_axis)
                # distance_list, angle_list = compute_and_visualize_distances(img, sorted_centers)
                # cnt_all[subject]['cnt_sutures']['ordered_center'] = sorted_centers
                # cnt_all[subject]['cnt_sutures']['center_distance'] = distance_list
                # cnt_all[subject]['cnt_sutures']['curvature_angles'] = angle_list

                add_numbered_bbx(img, sorted_centers, boxes, principal_axis)

                output_path = os.path.join(output_dir, f"{subject}.jpg")

                centers = data['cnt_vessels']['centers']
                boxes = data['cnt_vessels']['boxes']
                for center, box in zip(centers, boxes):
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
                cnt_all[subject]['img_visall'] = img
                cv2.imwrite(output_path, img)

    return cnt_all



def max_curvature_angle(cnt_all):
    return_dict = {}
    for subj, data in cnt_all.items():
        return_dict[subj] = np.max(np.array(data['cnt_sutures']['curvature_angles']))
    return return_dict

def get_aspect_ratio(box):
    lengths = get_side_lengths(box)
    long_side = max(lengths)
    short_side = min(lengths)
    aspect_ratio = long_side / short_side if short_side != 0 else 0
    return aspect_ratio

# ALI Aspect Ratio
def count_bounding_boxes_aspect_ratio(cnt_all, threshold=1.5):
    count_dict = {}
    for subj, data in cnt_all.items():
        count = 0
        for box in data['cnt_sutures']['boxes']:
            aspect_ratio = get_aspect_ratio(box)
            if aspect_ratio < threshold:
                count += 1
        count_dict[subj] = count
    return count_dict

def reorder_cnt_sutures(cnt_all):
    def project_and_sort(centers, principal_axis):
        center_mean = np.mean(centers, axis=0)
        projected_centers = np.dot(centers - center_mean, principal_axis)
        sorted_indices = np.argsort(projected_centers)
        return sorted_indices
    
    for subj_name, data in cnt_all.items():
        centers = np.array(data['cnt_sutures']['centers'])
        principal_axis = data['cnt_sutures']['pca_axis']
        
        if centers.shape[0] > 1:
            sorted_indices = project_and_sort(centers, principal_axis)
            data['cnt_sutures']['centers'] = [data['cnt_sutures']['centers'][i] for i in sorted_indices]
            data['cnt_sutures']['boxes'] = [data['cnt_sutures']['boxes'][i] for i in sorted_indices]
            data['cnt_sutures']['cnt_points'] = [data['cnt_sutures']['cnt_points'][i] for i in sorted_indices]
    
    return cnt_all



def vis_ali_5(cnt_all, df, n_mistake, output_dir, ALI_Mistakes, aspect_ratio_threshold=1.5, longest_side_threshold=0.1, target_height=640, vis_metric_flag=True):
    os.makedirs(output_dir, exist_ok=True)
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    TP_C = 0
    FP_C = 0
    FN_C = 0
    TN_C = 0
    def vis_metric(center, img, metric):
        txt_position = tuple([center[0] + 30, center[1] + 30])
        font_scale = 0.5
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        put_text_with_background(img, f'{metric:.3f}', txt_position, font_scale, thickness, font_type, text_color, bg_color)
        cv2.line(img, center, txt_position, (0, 0, 0), 2)

    def put_text_with_background(img, text, position, font_scale, thickness, font_type, text_color, bg_color=(0, 0, 0)):
        (text_width, text_height), baseline = cv2.getTextSize(text, font_type, font_scale, thickness)
        x, y = position
        #cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
        #cv2.putText(img, text, position, font_type, font_scale, text_color, thickness, cv2.LINE_AA)

    def get_aspect_ratio(box):
        side_lengths = np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1)
        aspect_ratio = max(side_lengths) / min(side_lengths)
        return aspect_ratio

    def get_longest_side(box):
        side_lengths = np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1)
        return max(side_lengths)

    def get_aligned_sides(principal_axis, box):
        side_vectors = [box[(i + 1) % 4] - box[i] for i in range(4)]
        side_lengths = [np.linalg.norm(vec) for vec in side_vectors]
        side_angles = [angle_between_vectors(principal_axis, vec) for vec in side_vectors]
        aligned_indices = [i for i, angle in enumerate(side_angles) if abs(angle) < 45 or abs(angle) > 135]
        aligned_sides = [side_lengths[i] for i in aligned_indices]
        return aligned_sides, aligned_indices

    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img'].copy()
        
        if 'cnt_vessels' not in data:
            print(f"Vessel not detected for subject {subj_name}. Skipping image.")
            continue

        centers = data['cnt_sutures']['centers']
        boxes = data['cnt_sutures']['boxes']
        principal_axis = data['cnt_sutures']['pca_axis']

        aspect_ratio_count = 0
        longest_side_excess_count = 0

        for center, box in zip(centers, boxes):
            aspect_ratio = get_aspect_ratio(box)
            longest_side = get_longest_side(box)
            vessel_box = data['cnt_vessels']['boxes'][0]
            aligned_vessel_sides, aligned_indices = get_aligned_sides(principal_axis, vessel_box)
            normalized_length = longest_side / max(aligned_vessel_sides)
            
            color = (0, 0, 255) if (aspect_ratio < aspect_ratio_threshold and normalized_length < longest_side_threshold) else (0, 255, 0)
            if aspect_ratio < aspect_ratio_threshold or normalized_length < longest_side_threshold:
                aspect_ratio_count += 1


            cv2.drawContours(img, [box], 0, color, 4)
            # if vis_metric_flag and (aspect_ratio < aspect_ratio_threshold or normalized_length < longest_side_threshold):
            #     vis_metric(center, img, normalized_length)

        # Draw the vessel bounding box in black
        cv2.drawContours(img, [vessel_box], 0, (0, 0, 0), 2)
        # Highlight the aligned sides of the vessel in red
        for idx in aligned_indices:
            pt1 = tuple(vessel_box[idx])
            pt2 = tuple(vessel_box[(idx + 1) % 4])
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

        # Crop the image around the bounding boxes
        img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
        
        # Resize the image to have the target height while maintaining the aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        img = cv2.resize(img, (target_width, target_height))

        # Add text with a black background
        missing_suture = 8 - len(centers)
        #text1 = f'Human Score: {ALI_score} {ALI_Mistakes[f"Mistake {n_mistake}"]}'
        text1 = f'Human Score: {ALI_score}'
        text2 = f'Algorithm Score: {aspect_ratio_count + longest_side_excess_count + missing_suture}'
        algorithm_score = aspect_ratio_count + longest_side_excess_count + missing_suture
        if ALI_score > 0 and algorithm_score > 0:
            TP += 1
        if ALI_score > 0 and algorithm_score == 0:
            FN += 1
        if ALI_score == 0 and algorithm_score > 0:
            FP += 1
        if ALI_score == 0 and algorithm_score == 0:
            TN += 1

        
        if ALI_score > 0 and algorithm_score == ALI_score:
            TP_C += 1
        if ALI_score > 0 and algorithm_score != ALI_score:
            FN_C += 1
        if ALI_score == 0 and algorithm_score > 0:
            FP_C += 1
        if ALI_score == 0 and algorithm_score == 0:
            TN_C += 1

        position1 = (10, 40)
        position2 = (10, 80)

        font_scale = 1.0
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        put_text_with_background(img, text1, position1, font_scale, thickness, font_type, text_color)
        put_text_with_background(img, text2, position2, font_scale, thickness, font_type, text_color)

        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Error 4')
    print(f'Binary precision: {precision}, recall: {recall},accuracy: {accuracy}')
    
    precision_C = TP_C / (TP_C + FP_C)
    recall_C = TP_C / (TP_C + FN_C)
    accuracy_C = (TP_C + TN_C) / (TP_C + FP_C + TN_C + FN_C)
    print(f'precision: {precision_C}, recall: {recall_C},accuracy: {accuracy_C}')

def vis_metric(center,img,metric):
    txt_position = tuple([center[0]+30,center[1]+30])
    font_scale = 0.5
    thickness = 1
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    put_text_with_background(img, f'{metric:.1f}', txt_position, font_scale, thickness, font_type, text_color, bg_color)
    cv2.line(img, center, txt_position, (0,0,0),2)

def vis_ali_3(cnt_all, df, n_mistake, output_dir, ALI_Mistakes, angle_threshold=15, target_height=640, vis_metric_flag=True):
    os.makedirs(output_dir, exist_ok=True)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    TP_C = 0
    FP_C = 0
    FN_C = 0
    TN_C = 0
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img'].copy()
        
        centers = data['cnt_sutures']['centers']
        boxes = data['cnt_sutures']['boxes']
        principal_axis = data['cnt_sutures']['pca_axis']
        center_mean = np.mean(centers, axis=0)
        
        extension_length = 100
        projected_centers = np.dot(centers - center_mean, principal_axis)
        min_proj = projected_centers.min()
        max_proj = projected_centers.max()
        point1 = center_mean + (min_proj - extension_length) * principal_axis
        point2 = center_mean + (max_proj + extension_length) * principal_axis

        angle_excess_count = 0
        for center, box in zip(centers, boxes):
            rect = cv2.minAreaRect(box)
            short_axis_start, short_axis_end, short_axis_vector = compute_short_axis(box, rect, principal_axis)
            angle = angle_between_vectors(short_axis_vector, principal_axis)
            angle = min(angle, 180 - angle)
            color = (0, 0, 255) if angle > angle_threshold else (0, 255, 0)
            if angle > angle_threshold:
                angle_excess_count += 1
                if vis_metric_flag:
                    vis_metric(center, img, angle)
                
            cv2.drawContours(img, [box], 0, color, 2)
            cv2.line(img, tuple(short_axis_start.astype(int)), tuple(short_axis_end.astype(int)), color=color, thickness=2)
            
        # Draw principal axis line
        draw_dashed_line(img, point1, point2, color=(255, 0, 0), thickness=2, dash_length=20)
        
        # Add text with a black background
        #text1 = f'Human Score: ' + str(ALI_score) + ' ' + ALI_Mistakes[f'Mistake {n_mistake}']
        text1 = f'Human Score: ' + str(ALI_score)
        text2 = f'Algorithm Score: {angle_excess_count}'
        
        if ALI_score > 0 and angle_excess_count > 0:
            TP += 1
        if ALI_score > 0 and angle_excess_count == 0:
            FN += 1
        if ALI_score == 0 and angle_excess_count > 0:
            FP += 1
        if ALI_score == 0 and angle_excess_count == 0:
            TN += 1
        
        if ALI_score > 0 and angle_excess_count == ALI_score:
            TP_C += 1
        if ALI_score > 0 and angle_excess_count != ALI_score:
            FN_C += 1
        if ALI_score == 0 and angle_excess_count > 0:
            FP_C += 1
        if ALI_score == 0 and angle_excess_count == 0:
            TN_C += 1
        # if ALI_score > 0 and angle_excess_count == ALI_score:
        #     TP += 1
        # if ALI_score > 0 and angle_excess_count != ALI_score:
        #     FN += 1
        # if ALI_score == 0 and angle_excess_count > 0:
        #     FP += 1
        # if ALI_score == 0 and angle_excess_count == 0:
        #     TN += 1

        position1 = (10, 40)
        position2 = (10, 80)

        font_scale = 1.0
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        if 'cnt_vessels' in data:
            cropped_img = crop_rotated_bbx(data['img'], data['cnt_vessels']['cnt_points'][0])
            img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
        
        # Resize the final image to have the target height while maintaining the aspect ratio
        aspect_ratio_img = img.shape[1] / img.shape[0]
        target_width_img = int(target_height * aspect_ratio_img)
        img_resized = cv2.resize(img, (target_width_img, target_height))

        # Resize the cropped image to match the height of the final image
        aspect_ratio_cropped = cropped_img.shape[1] / cropped_img.shape[0]
        target_width_cropped = int(target_height * aspect_ratio_cropped)
        cropped_resized = cv2.resize(cropped_img, (target_width_cropped, target_height))

        put_text_with_background(img_resized, text1, position1, font_scale, thickness, font_type, color)
        put_text_with_background(img_resized, text2, position2, font_scale, thickness, font_type, color)

        # Concatenate the resized final image and the cropped image
        concatenated_img = np.concatenate((img_resized, cropped_resized), axis=1)

        # Save the final image and the concatenated image
        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img_resized)
        
        concatenated_output_path = os.path.join(output_dir, f"{subj_name}_concatenated.jpg")
        cv2.imwrite(concatenated_output_path, concatenated_img)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Error 2')
    print(f'Binary precision: {precision}, recall: {recall},accuracy: {accuracy}')
    
    precision_C = TP_C / (TP_C + FP_C)
    recall_C = TP_C / (TP_C + FN_C)
    accuracy_C = (TP_C + TN_C) / (TP_C + FP_C + TN_C + FN_C)
    print(f'precision: {precision_C}, recall: {recall_C},accuracy: {accuracy_C}')


def put_text_with_background(img, text, position, font_scale, thickness, font_type, text_color, bg_color=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font_type, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, position, font_type, font_scale, text_color, thickness, cv2.LINE_AA)

def vis_ali_1(cnt_all, df, n_mistake, output_dir, ALI_Mistakes, curvature_angle_threshold=10, target_height=640, vis_metric_flag=True):
    os.makedirs(output_dir, exist_ok=True)
    subj_count = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    TP_C = 0
    FP_C = 0
    FN_C = 0
    TN_C = 0
    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img'].copy()
        
        centers = data['cnt_sutures']['centers']
        boxes = data['cnt_sutures']['boxes']

        curvature_excess_count = 0
        lines_to_draw = []

        # Initialize all lines as green
        for i, (center, box) in enumerate(zip(centers, boxes)):
            # Draw bounding box in green
            cv2.drawContours(img, [box], 0, (0, 0, 0), 2)
            
            if i > 0:
                previous_center = centers[i - 1]
                lines_to_draw.append((tuple(previous_center), tuple(center), (0, 255, 0)))

        # Check angles and update line colors
        for i in range(2, len(centers)):
            prev_prev_center = centers[i - 2]
            previous_center = centers[i - 1]
            current_center = centers[i]
            curvature_angle = angle_between_vectors(
                np.array(previous_center) - np.array(prev_prev_center),
                np.array(current_center) - np.array(previous_center)
            )
            if curvature_angle > curvature_angle_threshold:
                curvature_excess_count += 1
                # Update the color of the two lines forming the angle to red
                for j in range(len(lines_to_draw)):
                    if lines_to_draw[j][0] == tuple(prev_prev_center) and lines_to_draw[j][1] == tuple(previous_center):
                        lines_to_draw[j] = (lines_to_draw[j][0], lines_to_draw[j][1], (0, 0, 255))
                    if lines_to_draw[j][0] == tuple(previous_center) and lines_to_draw[j][1] == tuple(current_center):
                        lines_to_draw[j] = (lines_to_draw[j][0], lines_to_draw[j][1], (0, 0, 255))
                
                # Visualize the angle
                if vis_metric_flag:
                    vis_metric(previous_center, img, curvature_angle)
                
        # Draw all lines
        for line in lines_to_draw:
            draw_dashed_line(img, line[0], line[1], line[2], thickness=5)

        # Crop the image around the bounding boxes
        if 'cnt_vessels' in data:
            cropped_img = crop_rotated_bbx(data['img'], data['cnt_vessels']['cnt_points'][0])
            img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
        # Resize the final image to have the target height while maintaining the aspect ratio
        aspect_ratio_img = img.shape[1] / img.shape[0]
        target_width_img = int(target_height * aspect_ratio_img)
        img_resized = cv2.resize(img, (target_width_img, target_height))

        # Resize the cropped image to match the height of the final image
        aspect_ratio_cropped = cropped_img.shape[1] / cropped_img.shape[0]
        target_width_cropped = int(target_height * aspect_ratio_cropped)
        cropped_resized = cv2.resize(cropped_img, (target_width_cropped, target_height))

        

        # Add text with a black background
        #text1 = f'Human Score: ' + str(ALI_score) + ' ' + ALI_Mistakes[f'Mistake {n_mistake}']
        text1 = f'Human Score: ' + str(ALI_score) 
        text2 = f'Algorithm Score: {curvature_excess_count}'
        
        # if ALI_score > 0 and curvature_excess_count > 0:
        #     TP += 1
        # if ALI_score > 0 and curvature_excess_count == 0:
        #     FN += 1
        # if ALI_score == 0 and curvature_excess_count > 0:
        #     FP += 1
        # if ALI_score == 0 and curvature_excess_count == 0:
        #     TN += 1
        if ALI_score > 0 and curvature_excess_count == ALI_score:
            TP += 1
        if ALI_score > 0 and curvature_excess_count != ALI_score:
            FN += 1
        if ALI_score == 0 and curvature_excess_count > 0:
            FP += 1
        if ALI_score == 0 and curvature_excess_count == 0:
            TN += 1
        
        if ALI_score > 0 and curvature_excess_count == ALI_score:
            TP_C += 1
        if ALI_score > 0 and curvature_excess_count != ALI_score:
            FN_C += 1
        if ALI_score == 0 and curvature_excess_count > 0:
            FP_C += 1
        if ALI_score == 0 and curvature_excess_count == 0:
            TN_C += 1
        
        
        
        
        position1 = (10, 40)
        position2 = (10, 80)

        font_scale = 1.0
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        put_text_with_background(img_resized, text1, position1, font_scale, thickness, font_type, text_color)
        put_text_with_background(img_resized, text2, position2, font_scale, thickness, font_type, text_color)
        
        
        # Concatenate the resized final image and the cropped image
        concatenated_img = np.concatenate((img_resized, cropped_resized), axis=1)
        
        # Save the final image (imageA) and the concatenated image
        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img_resized)
        
        concatenated_output_path = os.path.join(output_dir, f"{subj_name}_concatenated.jpg")
        cv2.imwrite(concatenated_output_path, concatenated_img)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Error 1---------------------')
    print(f'Binary precision: {precision}, recall: {recall},accuracy: {accuracy}')
    
    precision_C = TP_C / (TP_C + FP_C)
    recall_C = TP_C / (TP_C + FN_C)
    accuracy_C = (TP_C + TN_C) / (TP_C + FP_C + TN_C + FN_C)
    print(f'precision: {precision_C}, recall: {recall_C},accuracy: {accuracy_C}')

def get_longest_side(box):
    lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
    return max(lengths)

def get_aligned_sides(principal_axis, box):
    vectors = [box[(i+1) % 4] - box[i] for i in range(4)]
    angles = [angle_between_vectors(principal_axis, vector) for vector in vectors]
    aligned_side_indices = sorted(range(4), key=lambda i: min(angles[i], 180 - angles[i]))[:2]
    aligned_lengths = [np.linalg.norm(vectors[i]) for i in aligned_side_indices]
    return aligned_lengths, aligned_side_indices


def vis_ali_4(cnt_all, df, n_mistake, output_dir, ALI_Mistakes, longest_side_threshold=0.1, target_height=640, vis_metric_flag=True):
    os.makedirs(output_dir, exist_ok=True)
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    TP_C = 0
    FP_C = 0
    FN_C = 0
    TN_C = 0
    def vis_metric(center, img, metric):
        txt_position = tuple([center[0] + 30, center[1] + 30])
        font_scale = 0.5
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        put_text_with_background(img, f'{metric:.3f}', txt_position, font_scale, thickness, font_type, text_color, bg_color)
        cv2.line(img, center, txt_position, (0, 0, 0), 2)

    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img'].copy()
        
        if 'cnt_vessels' not in data:
            print(f"Vessel not detected for subject {subj_name}. Skipping image.")
            continue

        centers = data['cnt_sutures']['centers']
        boxes = data['cnt_sutures']['boxes']
        principal_axis = data['cnt_sutures']['pca_axis']

        longest_side_excess_count = 0
        for center, box in zip(centers, boxes):
            longest_side = get_longest_side(box)
            vessel_box = data['cnt_vessels']['boxes'][0]
            aligned_vessel_sides, aligned_indices = get_aligned_sides(principal_axis, vessel_box)
            normalized_length = longest_side / max(aligned_vessel_sides)
            color = (0, 0, 255) if normalized_length > longest_side_threshold else (0, 255, 0)
            if normalized_length > longest_side_threshold:
                longest_side_excess_count += 1
            cv2.drawContours(img, [box], 0, color, 4)
            if vis_metric_flag and normalized_length > longest_side_threshold:
                vis_metric(center, img, normalized_length)

        # Draw the vessel bounding box in black
        cv2.drawContours(img, [vessel_box], 0, (0, 0, 0), 2)
        # Highlight the aligned sides of the vessel in red
        for idx in aligned_indices:
            pt1 = tuple(vessel_box[idx])
            pt2 = tuple(vessel_box[(idx + 1) % 4])
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

        # Crop the image around the bounding boxes
        cropped_img = crop_rotated_bbx(data['img'], data['cnt_vessels']['cnt_points'][0])
        img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])
        
        # Resize the final image to have the target height while maintaining the aspect ratio
        aspect_ratio_img = img.shape[1] / img.shape[0]
        target_width_img = int(target_height * aspect_ratio_img)
        img_resized = cv2.resize(img, (target_width_img, target_height))

        # Resize the cropped image to match the height of the final image
        aspect_ratio_cropped = cropped_img.shape[1] / cropped_img.shape[0]
        target_width_cropped = int(target_height * aspect_ratio_cropped)
        cropped_resized = cv2.resize(cropped_img, (target_width_cropped, target_height))

        # Add text with a black background
        #text1 = f'Human Score: ' + str(ALI_score) + ' ' + ALI_Mistakes[f'Mistake {n_mistake}']
        text1 = f'Human Score: ' + str(ALI_score)
        text2 = f'Algorithm Score: {longest_side_excess_count}'
        if ALI_score > 0 and longest_side_excess_count > 0:
            TP += 1
        if ALI_score > 0 and longest_side_excess_count == 0:
            FN += 1
        if ALI_score == 0 and longest_side_excess_count > 0:
            FP += 1
        if ALI_score == 0 and longest_side_excess_count == 0:
            TN += 1
        
        if ALI_score > 0 and longest_side_excess_count == ALI_score:
            TP_C += 1
        if ALI_score > 0 and longest_side_excess_count != ALI_score:
            FN_C += 1
        if ALI_score == 0 and longest_side_excess_count > 0:
            FP_C += 1
        if ALI_score == 0 and longest_side_excess_count == 0:
            TN_C += 1

        position1 = (10, 40)
        position2 = (10, 80)

        font_scale = 1.0
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        put_text_with_background(img_resized, text1, position1, font_scale, thickness, font_type, text_color)
        put_text_with_background(img_resized, text2, position2, font_scale, thickness, font_type, text_color)

        # Concatenate the resized final image and the cropped image
        concatenated_img = np.concatenate((img_resized, cropped_resized), axis=1)

        # Save the final image and the concatenated image
        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img_resized)
        
        concatenated_output_path = os.path.join(output_dir, f"{subj_name}_concatenated.jpg")
        cv2.imwrite(concatenated_output_path, concatenated_img)
        
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Error 3')
    print(f'Binary precision: {precision}, recall: {recall},accuracy: {accuracy}')
    
    precision_C = TP_C / (TP_C + FP_C)
    recall_C = TP_C / (TP_C + FN_C)
    accuracy_C = (TP_C + TN_C) / (TP_C + FP_C + TN_C + FN_C)
    print(f'precision: {precision_C}, recall: {recall_C},accuracy: {accuracy_C}')

def vis_ali_6(cnt_all, df, n_mistake, output_dir, ALI_Mistakes, max_thresh, min_thresh, target_height=640, vis_metric_flag=True):
    os.makedirs(output_dir, exist_ok=True)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    TP_C = 0
    FP_C = 0
    FN_C = 0
    TN_C = 0

    def project_point_on_axis(point, axis_origin, axis_direction):
        return np.dot(point - axis_origin, axis_direction) * axis_direction + axis_origin

    for subj_name, data in cnt_all.items():
        ALI_score = df.loc[df['Subject'] == subj_name, f'Mistake {n_mistake}'].values[0]
        img = data['img'].copy()
        
        if 'cnt_vessels' not in data:
            print(f"Vessel not detected for subject {subj_name}. Skipping image.")
            continue
        
        error_count = 0
        centers = data['cnt_sutures']['centers']
        boxes = data['cnt_sutures']['boxes']
        principal_axis = data['cnt_sutures']['pca_axis']
        center_mean = np.mean(centers, axis=0)
        
        vessel_box = data['cnt_vessels']['boxes'][0]
        aligned_vessel_sides, aligned_indices = get_aligned_sides(principal_axis, vessel_box)
        # distance_threshold = max(aligned_vessel_sides) / 8 * 1.5
        # distance_low_threshold = max(aligned_vessel_sides) / 8 * 0.5
        distance_threshold = max(aligned_vessel_sides) *max_thresh
        distance_low_threshold = max(aligned_vessel_sides) *  min_thresh
        

        for i, (center, box) in enumerate(zip(centers, boxes)):
            cv2.drawContours(img, [box], 0, (0, 0, 0), 2)

        for i in range(1, len(centers)):
            prev_center = centers[i - 1]
            curr_center = centers[i]

            prev_proj = project_point_on_axis(prev_center, center_mean, principal_axis)
            curr_proj = project_point_on_axis(curr_center, center_mean, principal_axis)
            projection_distance = np.linalg.norm(curr_proj - prev_proj)

            color = (0, 0, 255) if projection_distance > distance_threshold or projection_distance < distance_low_threshold else (0, 255, 0)
            if projection_distance > distance_threshold or projection_distance < distance_low_threshold:
                if projection_distance > distance_threshold:
                    print(subj_name,'too long', round(projection_distance /max(aligned_vessel_sides) *1000)/1000)
                else:
                    print(subj_name,'too short', round(projection_distance /max(aligned_vessel_sides) *1000)/1000)
                #print(subj_name,round(projection_distance /max(aligned_vessel_sides) *100)/100)
                error_count += 1

            draw_dashed_line(img, prev_center, curr_center, color, thickness=5,dash_length= 10)

        # Draw the vessel bounding box in black
        cv2.drawContours(img, [vessel_box], 0, (0, 0, 0), 2)
        # Highlight the aligned sides of the vessel in red
        for idx in aligned_indices:
            pt1 = tuple(vessel_box[idx])
            pt2 = tuple(vessel_box[(idx + 1) % 4])
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

        # Draw the principal axis line
        projected_centers = np.dot(centers - center_mean, principal_axis)
        min_proj = projected_centers.min()
        max_proj = projected_centers.max()
        point1 = center_mean + (min_proj - 100) * principal_axis
        point2 = center_mean + (max_proj + 100) * principal_axis
        #draw_dashed_line(img, point1, point2, color=(255, 0, 0), thickness=2)

        # Crop the image around the bounding boxes
        img = crop_rotated_bbx(img, data['cnt_vessels']['cnt_points'][0])

        # Resize the image to have the target height while maintaining the aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        img = cv2.resize(img, (target_width, target_height))

        # Add text with a black background
        text1 = 'Human Score: ' + str(ALI_score)
        text2 = f'Algorithm Score: {error_count}'
        
        if ALI_score > 0 and error_count > 0:
            TP += 1
        if ALI_score > 0 and error_count == 0:
            FN += 1
        if ALI_score == 0 and error_count > 0:
            FP += 1
        if ALI_score == 0 and error_count == 0:
            TN += 1
        
        if ALI_score > 0 and error_count == ALI_score:
            TP_C += 1
        if ALI_score > 0 and error_count != ALI_score:
            FN_C += 1
        if ALI_score == 0 and error_count > 0:
            FP_C += 1
        if ALI_score == 0 and error_count == 0:
            TN_C += 1

        position1 = (10, 40)
        position2 = (10, 80)

        font_scale = 0.7
        thickness = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        put_text_with_background(img, text1, position1, font_scale, thickness, font_type, text_color)
        put_text_with_background(img, text2, position2, font_scale, thickness, font_type, text_color)

        output_path = os.path.join(output_dir, f"{subj_name}.jpg")
        cv2.imwrite(output_path, img)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Error 5')
    print(f'Binary precision: {precision}, recall: {recall},accuracy: {accuracy}')
    
    precision_C = TP_C / (TP_C + FP_C)
    recall_C = TP_C / (TP_C + FN_C)
    accuracy_C = (TP_C + TN_C) / (TP_C + FP_C + TN_C + FN_C)
    print(f'precision: {precision_C}, recall: {recall_C},accuracy: {accuracy_C}')



def project_point_on_axis(point, axis_origin, axis_direction):
    return np.dot(point - axis_origin, axis_direction) * axis_direction + axis_origin

def calculate_accuracy(cnt_all, df_ALI, ALI_Error, threshold):
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for subj_name, data in cnt_all.items():
        human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
        algorithm_score = 0
        
        if ALI_Error == 1:
            max_angle = max(data['cnt_sutures']['curvature_angles'])
            algorithm_score = 1 if max_angle > threshold else 0
        elif ALI_Error == 3:
            max_angle = max(data['cnt_sutures']['angles_with_principal_axis'])
            algorithm_score = 1 if max_angle > threshold else 0
        elif ALI_Error == 4:
            longest_sides = [get_longest_side(box) for box in data['cnt_sutures']['boxes']]
            aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
            normalized_lengths = [side / max(aligned_vessel_sides) for side in longest_sides]
            algorithm_score = 1 if any(length > threshold for length in normalized_lengths) else 0
        elif ALI_Error == 5:
            aspect_ratios = [get_aspect_ratio(box) for box in data['cnt_sutures']['boxes']]
            algorithm_score = 1 if any(ratio < threshold for ratio in aspect_ratios) else 0
        
        elif ALI_Error == 6:
            centers = data['cnt_sutures']['centers']
            principal_axis = data['cnt_sutures']['pca_axis']
            center_mean = np.mean(centers, axis=0)
            aligned_vessel_sides, _ = get_aligned_sides(principal_axis, data['cnt_vessels']['boxes'][0])
            max_aligned_vessel_side = max(aligned_vessel_sides)
            
            distances = []
            for i in range(1, len(centers)):
                prev_center = centers[i - 1]
                curr_center = centers[i]
                prev_proj = project_point_on_axis(prev_center, center_mean, principal_axis)
                curr_proj = project_point_on_axis(curr_center, center_mean, principal_axis)
                projection_distance = np.linalg.norm(curr_proj - prev_proj)
                normalized_distance = projection_distance / max_aligned_vessel_side
                distances.append(normalized_distance)

            distance_scores = [(distance - mean_distance) ** 2 for distance in distances]
            human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
            if human_score > 0:
                metric_value = sorted(distance_scores, reverse=True)[:human_score]
            else:
                metric_value = max(distance_scores)
        
        if algorithm_score == 1 and human_score == 1:
            tp += 1
        elif algorithm_score == 0 and human_score == 0:
            tn += 1
        elif algorithm_score == 1 and human_score == 0:
            fp += 1
        elif algorithm_score == 0 and human_score == 1:
            fn += 1
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


from scipy.stats import norm
import seaborn as sns
from sklearn.metrics import roc_curve,auc

# def calculate_optimal_threshold(scores, labels):
#     fpr, tpr, thresholds = roc_curve(labels, scores)
    
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     return optimal_threshold



def find_best_min_max_thresholds(scores, labels, min_range, max_range):
    best_auc = 0
    best_min = None
    best_max = None

    for min_val in min_range:
        for max_val in max_range:
            if min_val >= max_val:
                continue
            
            # Generate predictions based on current min and max thresholds
            predictions = [not (min_val < score < max_val) for score in scores]
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            current_auc = auc(fpr, tpr)
            
            # Update best min, max and AUC if current AUC is better
            if current_auc > best_auc:
                best_auc = current_auc
                best_min = min_val
                best_max = max_val

    return best_min, best_max

def calculate_optimal_threshold(scores, labels, is_range=False, is_dual=False):
    if is_dual:
        longest_sides, aspect_ratios = zip(*scores)
        min_range1 = np.linspace(min(longest_sides), max(longest_sides), 100)
        min_range2 = np.linspace(min(aspect_ratios), max(aspect_ratios), 100)
        best_min1, best_min2 = find_best_min_max_thresholds_dual(longest_sides, aspect_ratios, labels, min_range1, min_range2)
        return (best_min1, best_min2)
    elif is_range:
        min_range = np.linspace(min(scores), max(scores), 100)
        max_range = np.linspace(min(scores), max(scores), 100)
        best_min, best_max = find_best_min_max_thresholds(scores, labels, min_range, max_range)
        return (best_min, best_max)
    else:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

def find_best_min_max_thresholds_dual(longest_sides, aspect_ratios, labels, min_range1, min_range2):
    best_auc = 0
    best_min1 = None
    best_min2 = None

    for min_val1 in min_range1:
        for min_val2 in min_range2:
            # Generate predictions based on current min thresholds
            predictions = [(side < min_val1) or (ar < min_val2) for side, ar in zip(longest_sides, aspect_ratios)]
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            current_auc = auc(fpr, tpr)
            
            # Update best min1, min2 and AUC if current AUC is better
            if current_auc > best_auc:
                best_auc = current_auc
                best_min1 = min_val1
                best_min2 = min_val2

    return best_min1, best_min2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from pathlib import Path
def five_fold_cross(scores, labels, save_dir, ALI_Error, optimal_threshold=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert labels to binary
    labels = np.array(labels, dtype=int)
    scores = np.array(scores)
    
    # Separate the true and false examples
    true_scores = scores[labels == 1]
    false_scores = scores[labels == 0]
    true_labels = labels[labels == 1]
    false_labels = labels[labels == 0]
    
    plt.figure(figsize=(10,1.5))
    #plt.figure(figsize=(10,0.5))
    # Compute and visualize the optimal threshold on the full data
    if ALI_Error == 5:
        longest_sides, aspect_ratios = zip(*scores)
        min_range1 = np.linspace(min(longest_sides), max(longest_sides), 100)
        min_range2 = np.linspace(min(aspect_ratios), max(aspect_ratios), 100)
        optimal_threshold1, optimal_threshold2 = find_best_min_max_thresholds_dual(longest_sides, aspect_ratios, labels, min_range1, min_range2)
        print(optimal_threshold1,optimal_threshold2)
        optimal_threshold = (optimal_threshold1, optimal_threshold2)
        
        
        longest_sides = np.array(longest_sides)
        aspect_ratios = np.array(aspect_ratios)
        
        # Plot points with label 0 (green)
        plt.scatter(longest_sides[labels == 0], aspect_ratios[labels == 0], color='green',s=100)
        plt.scatter(longest_sides[labels == 1], aspect_ratios[labels == 1], color='red',s=100)
        #plt.scatter(longest_sides, aspect_ratios, c=labels, cmap='RdYlGn', label='All Data')
        #plt.scatter(longest_sides[labels == 0], aspect_ratios[labels == 0], color='green')
        plt.axvline(x=optimal_threshold1, color='blue', linestyle='--', label=f'Threshold Longest Side: {optimal_threshold1:.2f}')
        plt.axhline(y=optimal_threshold2, color='orange', linestyle='--', label=f'Threshold Aspect Ratio: {optimal_threshold2:.2f}')
    
        plt.xlabel('Normalized stitch width',fontsize=20)
        #plt.ylabel('Aspect Ratio',fontsize=16)
        #plt.title('Error 4: Partial Thickness Stitch',fontsize=20)
        plt.xlim(left=0)
        plt.ylim(bottom=0.5,top=5.0)
        #plt.legend(fontsize=12)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(Path(save_dir).parent, f'full_data_{ALI_Error}.png'),dpi=800, transparent=True,bbox_inches='tight')
        plt.show()
        
        #plt.title(f'ALI {ALI_Error} Full Data - Optimal Threshold')
        #plt.colorbar(label='Label')
        plt.show()
    elif ALI_Error == 6:
        min_range = np.linspace(min(scores), max(scores), 100)
        max_range = np.linspace(min(scores), max(scores), 100)
        optimal_min, optimal_max = find_best_min_max_thresholds(scores, labels, min_range, max_range)
        optimal_threshold = (optimal_min, optimal_max)
        

        scores = np.array(scores)
        plt.scatter(true_scores, np.ones_like(true_scores), color='red',s=100)
        plt.scatter(false_scores, np.zeros_like(false_scores), color='green',s=100)
        plt.axvline(x=optimal_threshold[0], color='blue', linestyle='--', label=f'Min Threshold: {optimal_threshold[0]:.2f}')
        plt.axvline(x=optimal_threshold[1], color='blue', linestyle='--', label=f'Max Threshold: {optimal_threshold[1]:.2f}')
        print(optimal_threshold)
        plt.xlabel('Normalized inter-stitch distance',fontsize=20)
        #plt.title('Error 5: Unequal Distancing',fontsize=20)
        plt.xlim(left=-0.001)
        #plt.ylabel('Label',fontsize=20)
        plt.yticks([0,1])
        plt.ylim(-0.3,1.3)
    
        
        
        #plt.legend(fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(Path(save_dir).parent, f'full_data_{ALI_Error}.png'),dpi=800, transparent=True,bbox_inches='tight')
        plt.show()
    else:
        optimal_threshold = calculate_optimal_threshold(scores, labels)
        print(optimal_threshold)
        

        plt.scatter(true_scores, np.ones_like(true_scores), color='red',s=100)
        plt.scatter(false_scores, np.zeros_like(false_scores), color='green',s=100)
        plt.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
        #plt.title(f'ALI {ALI_Error} Full Data - Optimal Threshold')

        
        if ALI_Error == 1:
            plt.xlabel('Inter-Stitch curvature deviation (degree)',fontsize=20)
            #plt.ylabel('Label',fontsize=20)
            #plt.title('Error 1: Disruption of anastomosis line',fontsize=20,)
        
        if ALI_Error == 3:
            plt.xlabel('Stitch orientation deviation (degree)',fontsize=20)
            #plt.title('Error 2: Oblique Stitch',fontsize=20)
            #plt.ylabel('Label',fontsize=20)
        
        if ALI_Error == 4:
            plt.xlabel('Normalized stitch width',fontsize=20)
            #plt.title('Error 3: Too Wide Stitch',fontsize=20)
            #plt.ylabel('Label',fontsize=20)

        
        #plt.legend(fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.yticks([0,1])
        plt.ylim(-0.3,1.3)
        plt.tight_layout()
        plt.xlim(left=0)
        plt.savefig(os.path.join(Path(save_dir).parent, f'full_data_{ALI_Error}.png'),dpi=800, transparent=True,bbox_inches='tight')
        plt.show()
    
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    true_folds = list(skf.split(true_scores, true_labels))
    false_folds = list(skf.split(false_scores, false_labels))

    accuracies = []
    precisions = []
    recalls = []
    
    for fold_idx in range(5):
        true_train_idx, true_test_idx = true_folds[fold_idx]
        false_train_idx, false_test_idx = false_folds[fold_idx]
    
        train_scores = np.concatenate([true_scores[true_train_idx], false_scores[false_train_idx]])
        test_scores = np.concatenate([true_scores[true_test_idx], false_scores[false_test_idx]])
    
        train_labels = np.concatenate([true_labels[true_train_idx], false_labels[false_train_idx]])
        test_labels = np.concatenate([true_labels[true_test_idx], false_labels[false_test_idx]])
        
        # Find the optimal threshold on the training set
        if ALI_Error == 5:
            longest_sides_train, aspect_ratios_train = zip(*train_scores)
            min_range1 = np.linspace(min(longest_sides_train), max(longest_sides_train), 100)
            min_range2 = np.linspace(min(aspect_ratios_train), max(aspect_ratios_train), 100)
            optimal_threshold1, optimal_threshold2 = find_best_min_max_thresholds_dual(longest_sides_train, aspect_ratios_train, train_labels, min_range1, min_range2)
            optimal_threshold = (optimal_threshold1, optimal_threshold2)
        elif ALI_Error == 6:
            min_range = np.linspace(min(train_scores), max(train_scores), 100)
            max_range = np.linspace(min(train_scores), max(train_scores), 100)
            optimal_min, optimal_max = find_best_min_max_thresholds(train_scores, train_labels, min_range, max_range)
            optimal_threshold = (optimal_min, optimal_max)
        else:
            optimal_threshold = calculate_optimal_threshold(train_scores, train_labels)
    
        # Predict on the test set
        if ALI_Error == 5:
            longest_sides_test, aspect_ratios_test = zip(*test_scores)
            test_predictions = [(side < optimal_threshold[0]) or (ar < optimal_threshold[1]) for side, ar in zip(longest_sides_test, aspect_ratios_test)]
        elif ALI_Error == 6:
            test_predictions = [not(optimal_threshold[0] < score < optimal_threshold[1]) for score in test_scores]
        else:
            test_predictions = test_scores > optimal_threshold
    
        # Calculate accuracy and recall
        accuracy = accuracy_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions)
    
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)

        # Visualization of train and test data
        plt.figure(figsize=(12,4))
        if ALI_Error == 5:
            longest_sides_train, aspect_ratios_train = zip(*train_scores)
            longest_sides_test, aspect_ratios_test = zip(*test_scores)
            plt.scatter([x for x, label in zip(longest_sides_train, train_labels) if label == 1], [y for y, label in zip(aspect_ratios_train, train_labels) if label == 1], color='red', label='Train True')
            plt.scatter([x for x, label in zip(longest_sides_train, train_labels) if label == 0], [y for y, label in zip(aspect_ratios_train, train_labels) if label == 0], color='green', label='Train False')
            plt.scatter([x for x, label in zip(longest_sides_test, test_labels) if label == 1], [y for y, label in zip(aspect_ratios_test, test_labels) if label == 1], facecolors='none', edgecolors='red', marker='o', label='Test True')
            plt.scatter([x for x, label in zip(longest_sides_test, test_labels) if label == 0], [y for y, label in zip(aspect_ratios_test, test_labels) if label == 0], facecolors='none', edgecolors='green', marker='o', label='Test False')

            plt.axvline(x=optimal_threshold[0], color='blue', linestyle='--', label=f'Threshold Longest Side: {optimal_threshold[0]:.2f}')
            plt.axhline(y=optimal_threshold[1], color='orange', linestyle='--', label=f'Threshold Aspect Ratio: {optimal_threshold[1]:.2f}')
            plt.xlabel('Longest Side')
            plt.ylabel('Aspect Ratio')
        elif ALI_Error == 6:
            plt.scatter(train_scores[train_labels == 1], np.ones_like(train_scores[train_labels == 1]), color='red', label='Train True')
            plt.scatter(train_scores[train_labels == 0], np.zeros_like(train_scores[train_labels == 0]), color='green', label='Train False')
            plt.scatter(test_scores[test_labels == 1], np.ones_like(test_scores[test_labels == 1]) + 0.1, color='red', marker='x', label='Test True')
            plt.scatter(test_scores[test_labels == 0], np.zeros_like(test_scores[test_labels == 0]) - 0.1, color='green', marker='x', label='Test False')
            plt.axvline(x=optimal_threshold[0], color='blue', linestyle='--', label=f'Min Threshold: {optimal_threshold[0]:.2f}')
            plt.axvline(x=optimal_threshold[1], color='orange', linestyle='--', label=f'Max Threshold: {optimal_threshold[1]:.2f}')
        else:
            plt.scatter(train_scores[train_labels == 1], np.ones_like(train_scores[train_labels == 1]), color='red', label='Train True')
            plt.scatter(train_scores[train_labels == 0], np.zeros_like(train_scores[train_labels == 0]), color='green', label='Train False')
            plt.scatter(test_scores[test_labels == 1], np.ones_like(test_scores[test_labels == 1]) + 0.1, color='red', edgecolors='k', marker='x', label='Test True')
            plt.scatter(test_scores[test_labels == 0], np.zeros_like(test_scores[test_labels == 0]) - 0.1, color='green', edgecolors='k', marker='x', label='Test False')
            plt.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
            #plt.xlabel('Scores')
            #plt.ylabel('Label')
            
        plt.title(f'ALI {ALI_Error} Fold {fold_idx + 1} - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}')
        plt.legend()

        plt.savefig(os.path.join(save_dir, f'fold_{fold_idx + 1}_threshold.png'))
        plt.show()
    
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_precision = np.mean(precision)
    std_precision = np.std(precision)
    print('---------------------------')
    print(f"Mean Accuracy: {mean_accuracy:.3f}, Std Accuracy: {std_accuracy:.3f}")
    print(f"Mean Recall: {mean_recall:.3f}, Std Recall: {std_recall:.3f}")
    print(f"Mean Precision: {mean_precision:.3f}, Std Precision: {std_precision:.3f}")



def save_roc(scores,labels,save_dir, ALI_score):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic of ALI Error {ALI_score}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir,'roc.png'))
    

def calculate_metric(cnt_all, ALI_Error, df_ALI, threshold, save_dir):
    metrics = []
    all_scores = []
    all_labels = []
    index_name_map = {}
    for subj_name, data in cnt_all.items():
        metric_value = 0
        
        if ALI_Error == 1:
            # if len(data['cnt_sutures']['curvature_angles']) < 4:
            #     continue  # Skip subjects with less than 5 sutures
            metric_value = max(data['cnt_sutures']['curvature_angles'])

            
        elif ALI_Error == 3:
            human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
            if human_score > 0:
                metric_value = sorted(data['cnt_sutures']['angles_with_principal_axis'], reverse=True)[:human_score]
            else:
                metric_value = max(data['cnt_sutures']['angles_with_principal_axis'])
            print(subj_name,metric_value)   
        elif ALI_Error == 4:
            longest_sides = [get_longest_side(box) for box in data['cnt_sutures']['boxes']]
            aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
            normalized_lengths = [side / max(aligned_vessel_sides) for side in longest_sides]
            metric_value = max(normalized_lengths)
            human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
            if human_score > 0:
                metric_value = sorted(normalized_lengths, reverse=True)[:human_score]
            else:
                metric_value = max(normalized_lengths)
            
        
        
        elif ALI_Error == 5:
            aspect_ratios = [get_aspect_ratio(box) for box in data['cnt_sutures']['boxes']]
            longest_sides = [get_longest_side(box) for box in data['cnt_sutures']['boxes']]
            aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
            normalized_longest_sides = [(0.08549865318073842 - side/max(aligned_vessel_sides)) / 0.03011157654627571 for side in longest_sides]
            normalized_aspect_ratios = [(3.515864051226026 - ar) / 1.3016186966728807 for ar in aspect_ratios]
            
            # Calculate combined score
            combined_scores = [(nls + nar) for nls, nar in zip(normalized_longest_sides, normalized_aspect_ratios)]
            
            human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
            num_missing_stitches = 8 - len(combined_scores)
            if human_score == 0 or human_score - num_missing_stitches <= 0:
                metric_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[0]
                metric_value = [(longest_sides[metric_indices]/max(aligned_vessel_sides), aspect_ratios[metric_indices])]
            elif human_score > 0 and human_score - num_missing_stitches > 0:
                
                #metric_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:human_score - num_missing_stitches]
                metric_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:human_score - num_missing_stitches]
                metric_value = [(longest_sides[i]/max(aligned_vessel_sides), aspect_ratios[i]) for i in metric_indices]

            
        elif ALI_Error == 6:
            centers = data['cnt_sutures']['centers']
            aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
            max_aligned_vessel_side = max(aligned_vessel_sides)
            
            distances = []
            

            boxes = data['cnt_sutures']['boxes']
            principal_axis = data['cnt_sutures']['pca_axis']
            center_mean = np.mean(centers, axis=0)
            
            accu_proj = 0
            for i in range(1, len(centers)):
                prev_center = centers[i - 1]
                curr_center = centers[i]
                distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                normalized_distance = distance / max_aligned_vessel_side
                #distances.append(normalized_distance)
                
                prev_proj = project_point_on_axis(prev_center, center_mean, principal_axis)
                curr_proj = project_point_on_axis(curr_center, center_mean, principal_axis)
                projection_distance = np.linalg.norm(curr_proj - prev_proj) / max_aligned_vessel_side
                distances.append(projection_distance)
                accu_proj += projection_distance
                #print(normalized_distance,projection_distance)
            expected_proj = accu_proj / (len(centers) - 1)
            human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
            if human_score > 0:
                metric_value = sorted(distances, key=lambda x: abs(x -expected_proj), reverse=True)[:human_score]
  
            else:
                metric_value = sorted(distances, key=lambda x: abs(x - expected_proj), reverse=True)[0]
        
        human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
        if isinstance(metric_value,list):
            all_scores.extend(metric_value)
        else:
            all_scores.append(metric_value)
        if isinstance(metric_value,list) and human_score >0:
            all_labels.extend([True] * len(metric_value))
        else:
            all_labels.append(human_score>0)


        metrics.append((subj_name, metric_value))
    
    
    np.save(f'plot_data/threshold/Error_{ALI_Error}_values.npy',all_scores)
    np.save(f'plot_data/threshold/Error_{ALI_Error}_labels.npy',all_labels)
    
    
    if ALI_Error == 6:
        optimal_threshold = calculate_optimal_threshold(all_scores, all_labels, is_range=True)
    elif ALI_Error == 5:
        optimal_threshold = calculate_optimal_threshold(all_scores, all_labels, is_dual=True)
    else:
        optimal_threshold = calculate_optimal_threshold(all_scores, all_labels)
    five_fold_cross(all_scores, all_labels,save_dir,ALI_Error)
    if ALI_Error != 5:
        save_roc(all_scores,all_labels,save_dir,ALI_Error)
    

    return metrics, optimal_threshold

def plot_distribution_and_fit_gaussian(metrics, df_ALI, cnt_all, ALI_Error,optimal_threshold, save_path):
    scores_1 = []
    scores_0 = []
    names_1 = []
    names_0 = []
    
    for subj_name, metric_value in metrics:
        human_score = df_ALI.loc[df_ALI['Subject'] == subj_name, f'Mistake {ALI_Error}'].values[0]
        data = cnt_all[subj_name]
        
        # if human_score >1:
        #     scores_1.extend(metric_value)
        #     names_1.extend([subj_name] * len(metric_value))
        # elif human_score  == 1:
        #     if isinstance(metric_value, list):
        #         scores_1.extend(metric_value)
        #         names_1.append(subj_name)
        #     else:
        #         scores_1.append(metric_value)
        #         names_1.append(subj_name)
        # else:
        #     scores_0.append(metric_value)
        #     names_0.append(subj_name)
        
        if isinstance(metric_value, list):
            if human_score > 0:
                scores_1.extend(metric_value)
                names_1.extend([subj_name] * len(metric_value))
            else:
                scores_0.extend(metric_value)
                names_0.extend([subj_name] * len(metric_value))
        
        else:
            if human_score > 0:
                scores_1.append(metric_value)
                names_1.append(subj_name)
            else:
                scores_0.append(metric_value)
                names_0.append(subj_name)
            
        
        
    
    # Determine common bin edges for uniform bin widths
    all_scores = scores_1 + scores_0
    min_score = min(all_scores)
    max_score = max(all_scores)
    bin_edges = np.linspace(min_score, max_score, 31)  # 20 bins
    
    # Plot histogram for human score 1
    plt.figure(figsize=(12, 6))
    sns.histplot(scores_1, bins=bin_edges, kde=False, color='red', label='Human Score 1', stat='count')
    
    # Plot histogram for human score 0
    sns.histplot(scores_0, bins=bin_edges, kde=False, color='green', label='Human Score 0', stat='count')
    
    # Add threshold line

    #optimal_threshold = calculate_optimal_threshold([m[1] for m in metrics], [df_ALI.loc[df_ALI['Subject'] == m[0], f'Mistake {ALI_Error}'].values[0] for m in metrics])
    if ALI_Error == 6:
        #print(optimal_threshold,ALI_Error)
        plt.axvline(x=optimal_threshold[0], color='blue', linestyle='--', label=f'Threshold: {optimal_threshold[0]:.2f}')
        plt.axvline(x=optimal_threshold[1], color='blue', linestyle='--', label=f'Threshold: {optimal_threshold[1]:.2f}')
        #plt.title(f'ALI {ALI_Error} Full Data - Optimal Threshold: {optimal_threshold[0]:.2f},{optimal_threshold[1]:.2f}')
    else:
        plt.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
        #plt.title(f'ALI {ALI_Error} Full Data - Optimal Threshold: {optimal_threshold:.2f}')
    
    # Add subject names on top of each histogram bar
    counts_1, _ = np.histogram(scores_1, bins=bin_edges)
    counts_0, _ = np.histogram(scores_0, bins=bin_edges)
    
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    max_count = max(counts_1 + counts_0)
    
    # Plot names for scores_1
    for i, count in enumerate(counts_1):
        bin_center = bin_centers[i]
        matching_names = [names_1[j] for j, score in enumerate(scores_1) if bin_edges[i] <= score < bin_edges[i+1]]
        if i == len(counts_1) - 1:  # Include the right edge for the last bin
            matching_names = [names_1[j] for j, score in enumerate(scores_1) if bin_edges[i] <= score <= bin_edges[i+1]]
        for j, name in enumerate(matching_names):
            name = name[7:]
            plt.text(bin_center, count + 0.02 * max_count * (j + 1), name, color='red', ha='center', fontsize=8)
    
    # Plot names for scores_0 with a vertical offset
    for i, count in enumerate(counts_0):
        bin_center = bin_centers[i]
        matching_names = [names_0[j] for j, score in enumerate(scores_0) if bin_edges[i] <= score < bin_edges[i+1]]
        if i == len(counts_0) - 1:  # Include the right edge for the last bin
            matching_names = [names_0[j] for j, score in enumerate(scores_0) if bin_edges[i] <= score <= bin_edges[i+1]]
        for j, name in enumerate(matching_names):
            name = name[7:]
            plt.text(bin_center, count + 0.02 * max_count * (j + 1) + 0.02 * max_count, name, color='green', ha='center', fontsize=8)
    
    plt.title(f'Distribution for ALI Error {ALI_Error}')
    plt.xlabel('Metric Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'0A_distribution_{ALI_Error}.png'))
    plt.show()




def calculate_and_plot_distributions_with_gaussian(cnt_all, output_dir):
    
    def project_point_on_axis(point, axis_origin, axis_direction):
        return np.dot(point - axis_origin, axis_direction)
    
    aspect_ratios = []
    normalized_longest_sides = []
    normalized_stitch_distances = []

    for subject, data in cnt_all.items():
        suture_boxes = data['cnt_sutures']['boxes']
        vessel_box = data['cnt_vessels']['boxes'][0]
        principal_axis = data['cnt_sutures']['pca_axis']

        aligned_vessel_sides, _ = get_aligned_sides(principal_axis, vessel_box)
        max_aligned_vessel_side = max(aligned_vessel_sides)

        # Calculate aspect ratios and normalized longest sides
        for suture_box in suture_boxes:
            aspect_ratio = get_aspect_ratio(suture_box)
            longest_side = get_longest_side(suture_box)
            normalized_longest_side = longest_side / max_aligned_vessel_side

            aspect_ratios.append(aspect_ratio)
            normalized_longest_sides.append(normalized_longest_side)

        # Calculate normalized stitch distances
        centers = data['cnt_sutures']['centers']
        center_mean = np.mean(centers, axis=0)
        for i in range(1, len(centers)):
            prev_center = centers[i - 1]
            curr_center = centers[i]
            prev_proj = project_point_on_axis(prev_center, center_mean, principal_axis)
            curr_proj = project_point_on_axis(curr_center, center_mean, principal_axis)
            projection_distance = np.linalg.norm(curr_proj - prev_proj)
            normalized_distance = projection_distance / max_aligned_vessel_side
            normalized_stitch_distances.append(normalized_distance)

    # Plotting the distributions with Gaussian fits
    plt.figure(figsize=(18, 6))

    # Aspect Ratios
    plt.subplot(1, 3, 1)
    sns.histplot(aspect_ratios, bins=30, kde=False, color='blue', stat='density')
    mu, std = norm.fit(aspect_ratios)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    print('Aspect Ratio',mu,std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.axvline(mu, color='r', linestyle='--', label='Mean')
    plt.axvline(mu + std, color='g', linestyle='--', label='1 STD')
    plt.axvline(mu - std, color='g', linestyle='--')
    plt.title('Distribution of Aspect Ratios')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Density')
    plt.legend()

    # Normalized Longest Sides
    plt.subplot(1, 3, 2)
    sns.histplot(normalized_longest_sides, bins=30, kde=False, color='green', stat='density')
    mu, std = norm.fit(normalized_longest_sides)
    print('Longest Side',mu,std)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.axvline(mu, color='r', linestyle='--', label='Mean')
    plt.axvline(mu + std, color='g', linestyle='--', label='1 STD')
    plt.axvline(mu - std, color='g', linestyle='--')
    
    plt.title('Distribution of Normalized Longest Sides')
    plt.xlabel('Normalized Longest Side')
    plt.ylabel('Density')
    plt.legend()

    # Normalized Stitch Distances
    plt.subplot(1, 3, 3)
    sns.histplot(normalized_stitch_distances, bins=30, kde=False, color='orange', stat='density')
    mu, std = norm.fit(normalized_stitch_distances)
    print('Stitch Distance',mu,std)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.axvline(mu, color='r', linestyle='--', label='Mean')
    plt.axvline(mu + std, color='g', linestyle='--', label='1 STD')
    plt.axvline(mu - std, color='g', linestyle='--')
    plt.title('Distribution of Normalized Stitch Distances')
    plt.xlabel('Normalized Stitch Distance')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_with_gaussian.png'))
    plt.show()

#   raw_dashed_line(img, point1, point2, color, thickness, dash_length=20)

def extract_ali_metrics(cnt_all):
    ali_metrics = {
        'ALI_Error_1_Angles': {},
        'ALI_Error_3_Angles': {},
        'ALI_Error_4_Normalized_Lengths': {},
        'ALI_Error_5_Aspect_Ratios': {},
        'ALI_Error_5_Normalized_Lengths': {},
        'ALI_Error_6_Distances': {}
    }

    for subj_name, data in cnt_all.items():
        # Extract angles for ALI Error 1
        ali_metrics['ALI_Error_1_Angles'][subj_name] = data['cnt_sutures']['curvature_angles']

        # Extract angles for ALI Error 3
        ali_metrics['ALI_Error_3_Angles'][subj_name] = data['cnt_sutures']['angles_with_principal_axis']

        # Extract normalized lengths for ALI Error 4
        longest_sides = [get_longest_side(box) for box in data['cnt_sutures']['boxes']]
        aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
        normalized_lengths = [side / max(aligned_vessel_sides) for side in longest_sides]
        ali_metrics['ALI_Error_4_Normalized_Lengths'][subj_name] = normalized_lengths
        if np.max(normalized_lengths) > 0.15:
            print('too wide stitch',subj_name,np.max(normalized_lengths))

        # Extract aspect ratios for ALI Error 5
        aspect_ratios = [get_aspect_ratio(box) for box in data['cnt_sutures']['boxes']]
        ali_metrics['ALI_Error_5_Aspect_Ratios'][subj_name] = aspect_ratios
        
        longest_sides = [get_longest_side(box) for box in data['cnt_sutures']['boxes']]
        aligned_vessel_sides, _ = get_aligned_sides(data['cnt_sutures']['pca_axis'], data['cnt_vessels']['boxes'][0])
        normalized_lengths = [side / max(aligned_vessel_sides) for side in longest_sides]
        ali_metrics['ALI_Error_5_Normalized_Lengths'][subj_name] = normalized_lengths

        # Extract distances for ALI Error 6
        centers = data['cnt_sutures']['centers']
        principal_axis = data['cnt_sutures']['pca_axis']
        center_mean = np.mean(centers, axis=0)
        aligned_vessel_sides, _ = get_aligned_sides(principal_axis, data['cnt_vessels']['boxes'][0])
        max_aligned_vessel_side = max(aligned_vessel_sides)

        distances = []

        for i in range(1, len(centers)):
            prev_center = centers[i - 1]
            curr_center = centers[i]
            distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
            normalized_distance = distance / max_aligned_vessel_side
            if subj_name == 'ClujDecS1':
                print(distance)
            distances.append(normalized_distance)

        ali_metrics['ALI_Error_6_Distances'][subj_name] = distances

    return ali_metrics

def visualize_ali_metrics_distribution(ali_metrics):
    # Set up the plotting environment
    sns.set(style="whitegrid")

    # Function to plot histogram and box plot
    def plot_distribution(data, title, xlabel):
        plt.figure(figsize=(14, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data, bins=30, kde=True, color='blue')
        plt.title(f'{title} - Histogram')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')

        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data, color='lightblue')
        plt.title(f'{title} - Box Plot')
        plt.xlabel(xlabel)

        plt.tight_layout()
        plt.show()

    # Extract data for plotting
    ali_error_1_angles = [angle for angles in ali_metrics['ALI_Error_1_Angles'].values() for angle in angles]
    ali_error_3_angles = [angle for angles in ali_metrics['ALI_Error_3_Angles'].values() for angle in angles]
    ali_error_4_lengths = [length for lengths in ali_metrics['ALI_Error_4_Normalized_Lengths'].values() for length in lengths]
    ali_error_5_aspect_ratios = [aspect_ratio for aspect_ratios in ali_metrics['ALI_Error_5_Aspect_Ratios'].values() for aspect_ratio in aspect_ratios]
    ali_error_6_distances = [distance for distances in ali_metrics['ALI_Error_6_Distances'].values() for distance in distances]

    # Plot distributions
    plot_distribution(ali_error_1_angles, 'ALI Error 1: Curvature Angles', 'Curvature Angle (degrees)')
    plot_distribution(ali_error_3_angles, 'ALI Error 3: Angles with Principal Axis', 'Angle with Principal Axis (degrees)')
    plot_distribution(ali_error_4_lengths, 'ALI Error 4: Normalized Lengths', 'Normalized Length')
    plot_distribution(ali_error_5_aspect_ratios, 'ALI Error 5: Aspect Ratios', 'Aspect Ratio')
    plot_distribution(ali_error_6_distances, 'ALI Error 6: Normalized Distances', 'Normalized Distance')


def compute_projections_vertical_distances_stitch_lengths(cnt_all):
    ali_metrics = {
        'ALI_Error_6_Projections': {},
        'ALI_Error_6_Vertical_Distances': {},
        'ALI_Error_6_Stitch_Length_Projections': {}
    }

    for subj_name, data in cnt_all.items():
        centers = data['cnt_sutures']['centers']
        principal_axis = data['cnt_sutures']['pca_axis']
        center_mean = np.mean(centers, axis=0)

        # Get the aligned vessel side length for normalization
        aligned_vessel_sides, _ = get_aligned_sides(principal_axis, data['cnt_vessels']['boxes'][0])
        max_aligned_vessel_side = max(aligned_vessel_sides)

        projections = []
        vertical_distances = []
        stitch_length_projections = []

        for i in range(1, len(centers)):
            prev_center = np.array(centers[i - 1])
            curr_center = np.array(centers[i])

            # Project the centers onto the principal axis
            prev_proj_scalar = np.dot(prev_center - center_mean, principal_axis)
            curr_proj_scalar = np.dot(curr_center - center_mean, principal_axis)

            # Convert scalars back to vector projections on the principal axis
            prev_proj = prev_proj_scalar * principal_axis + center_mean
            curr_proj = curr_proj_scalar * principal_axis + center_mean

            # Compute the projection distance (p)
            projection_distance = np.linalg.norm(curr_proj - prev_proj)
            normalized_projection_distance = projection_distance / max_aligned_vessel_side
            projections.append(normalized_projection_distance)
            if normalized_projection_distance > 0.3:
                print('too long horizontal',subj_name)
            
            # Compute the perpendicular (vertical) distance (h)
            vertical_distance = np.linalg.norm(curr_center - curr_proj)
            normalized_vertical_distance = vertical_distance / max_aligned_vessel_side
            vertical_distances.append(normalized_vertical_distance)
            if normalized_vertical_distance > 0.05:
                print('too wide vertical',subj_name)




        for box in data['cnt_sutures']['boxes']:
            # Compute the vectors of the sides of the bounding box
            side_vectors = box - np.roll(box, 1, axis=0)
            side_lengths = np.linalg.norm(side_vectors, axis=1)
            
            # Find the longest side vector
            longest_side_index = np.argmax(side_lengths)
            longest_side_vector = side_vectors[longest_side_index]

            # Project the longest side vector onto the principal axis
            stitch_length_projection = np.dot(longest_side_vector, principal_axis)
            normalized_stitch_length_projection = stitch_length_projection / max_aligned_vessel_side
            stitch_length_projections.append(np.abs(normalized_stitch_length_projection))
            if np.abs(normalized_stitch_length_projection) > 0.06:
                print('too obqiue stitch',subj_name,normalized_stitch_length_projection)
        ali_metrics['ALI_Error_6_Projections'][subj_name] = projections
        ali_metrics['ALI_Error_6_Vertical_Distances'][subj_name] = vertical_distances
        ali_metrics['ALI_Error_6_Stitch_Length_Projections'][subj_name] = stitch_length_projections

    return ali_metrics


def plot_distributions_by_expertise(ali_metrics, df_ali_scores, expertise_levels):
    for expertise in expertise_levels:
        subjects = df_ali_scores[df_ali_scores['Expertise'] == expertise]['Subject'].tolist()
        print(expertise,subjects)
        stitch_lengths_error_4 = [length for subj in subjects for length in ali_metrics['ALI_Error_4_Normalized_Lengths'].get(subj, [])]
        stitch_distance_along = [length for subj in subjects for length in ali_metrics['ALI_Error_6_Projections'].get(subj, [])]
        stitch_distance_vertical = [length for subj in subjects for length in ali_metrics['ALI_Error_6_Vertical_Distances'].get(subj, [])]
        stitch_length_projections = [length for subj in subjects for length in ali_metrics['ALI_Error_6_Stitch_Length_Projections'].get(subj, [])]
        
        # Plotting the distributions
        plt.figure(figsize=(20, 12))
        
        plt.subplot(2, 2, 1)
        sns.histplot(stitch_lengths_error_4, kde=True, bins=30, color='blue')
        plt.title(f'Distribution of Stitch Lengths (ALI Error 4) - {expertise}')
        plt.xlabel('Stitch Lengths')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 2)
        sns.histplot(stitch_distance_along, kde=True, bins=30, color='green')
        plt.title(f'Distribution of Stitch Distance Along (ALI Error 6 Projections) - {expertise}')
        plt.xlabel('Stitch Distance Along')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 3)
        sns.histplot(stitch_distance_vertical, kde=True, bins=30, color='red')
        plt.title(f'Distribution of Stitch Distance Vertical (ALI Error 6 Vertical Distances) - {expertise}')
        plt.xlabel('Stitch Distance Vertical')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 4)
        sns.histplot(stitch_length_projections, kde=True, bins=30, color='purple')
        plt.title(f'Distribution of Stitch Length Projections (ALI Error 6 Stitch Length Projections) - {expertise}')
        plt.xlabel('Stitch Length Projections')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

# Function to plot distributions for both expertise groups side by side
def plot_distributions_up_down(ali_metrics, df_ali_scores, expertise1, expertise2):
    subjects1 = df_ali_scores[df_ali_scores['Expertise'] == expertise1]['Subject'].tolist()
    subjects2 = df_ali_scores[df_ali_scores['Expertise'] == expertise2]['Subject'].tolist()
    
    metrics = [
        ('ALI_Error_4_Normalized_Lengths', 'Stitch Lengths (ALI Error 4)'),
        ('ALI_Error_6_Projections', 'Stitch Distance Along (ALI Error 6 Projections)'),
        ('ALI_Error_6_Vertical_Distances', 'Stitch Distance Vertical (ALI Error 6 Vertical Distances)'),
        ('ALI_Error_6_Stitch_Length_Projections', 'Stitch Length Projections (ALI Error 6 Stitch Length Projections)')
    ]
    
    
    
    for idx, (metric, title) in enumerate(metrics, 1):
        plt.figure(figsize=(12, 24))
        data1 = [length for subj in subjects1 for length in ali_metrics[metric].get(subj, [])]
        data2 = [length for subj in subjects2 for length in ali_metrics[metric].get(subj, [])]
        
        all_data = data1 + data2
        x_min, x_max = min(all_data), max(all_data)
        
        plt.subplot(2, 1, 1)
        sns.histplot(data1, kde=True, bins=30, color='red', stat='density')
        plt.title(f'{title} - {expertise1}')
        plt.xlabel(title)
        plt.ylabel('Density')
        plt.xlim(x_min, x_max)

        plt.subplot(2, 1, 2)
        sns.histplot(data2, kde=True, bins=30, color='green', stat='density')
        plt.title(f'{title} - {expertise2}')
        plt.xlabel(title)
        plt.ylabel('Density')
        plt.xlim(x_min, x_max)
    
    #plt.tight_layout()
    plt.show()
# Function to plot distributions for both expertise groups one above the other
# Function to plot distributions for all expertise groups one above the other
def plot_distributions_with_gaussian(ali_metrics, df_ali_scores, expertise1, expertise2, expertise3):
    subjects1 = df_ali_scores[df_ali_scores['Expertise'] == expertise1]['Subject'].tolist()
    subjects2 = df_ali_scores[df_ali_scores['Expertise'] == expertise2]['Subject'].tolist()
    subjects3 = df_ali_scores[df_ali_scores['Expertise'] == expertise3]['Subject'].tolist()
    print(subjects1, subjects2, subjects3)
    
    metrics = [
        ('ALI_Error_4_Normalized_Lengths', 'Distribution of Stitch Lengths (ALI 4)', 'Stitch Lengths'),
        ('ALI_Error_6_Projections', 'Distribution of Stitch Distance Projections (ALI 6)', 'Stitch Distance Projections'),
        ('ALI_Error_6_Vertical_Distances', 'Distribution of Stitch Distance Projection Errors (ALI 1)', 'Stitch Distance Projection Errors'),
        ('ALI_Error_6_Stitch_Length_Projections', 'Distribution of Stitch Length Projection Errors (ALI 3)', 'Stitch Length Projection Errors')
    ]
    
    for idx, (metric, title, xlabel) in enumerate(metrics, 1):
        
        fig, axs = plt.subplots(3, 1, figsize=(16, 10))
        
        data1 = [length for subj in subjects1 for length in ali_metrics[metric].get(subj, [])]
        data2 = [length for subj in subjects2 for length in ali_metrics[metric].get(subj, [])]
        data3 = [length for subj in subjects3 for length in ali_metrics[metric].get(subj, [])]
        
        all_data = data1 + data2 + data3
        x_min, x_max = min(all_data), max(all_data)
        
        # Plot for expertise1
        plt.subplot(3, 1, 1)
        sns.histplot(data1, kde=False, bins=30, color='red', stat='density')
        mean1, std1 = norm.fit(data1)
        x = np.linspace(x_min, x_max, 100)
        p = norm.pdf(x, mean1, std1)
        plt.plot(x, p, 'k', linewidth=2)
        plt.axvline(mean1, color='black', linestyle='--', label=f'Mean: {mean1:.4f}')
        plt.axvline(mean1 + std1, color='black', linestyle='--', label=f'STD: {std1:.4f}')
        plt.axvline(mean1 - std1, color='black', linestyle='--')
        plt.title(f'{title} - {expertise1}')
        #plt.xlabel(xlabel)
        plt.ylabel('Stitch Num')
        plt.xlim(x_min, x_max)
        plt.legend()

        # Plot for expertise2
        plt.subplot(3, 1, 2)
        sns.histplot(data2, kde=False, bins=30, color='green', stat='density')
        mean2, std2 = norm.fit(data2)
        x = np.linspace(x_min, x_max, 100)
        p = norm.pdf(x, mean2, std2)
        plt.plot(x, p, 'k', linewidth=2)
        plt.axvline(mean2, color='black', linestyle='--', label=f'Mean: {mean2:.4f}')
        plt.axvline(mean2 + std2, color='black', linestyle='--', label=f'STD: {std2:.4f}')
        plt.axvline(mean2 - std2, color='black', linestyle='--')
        plt.title(f'{title} - {expertise2}')
        #plt.xlabel(xlabel)
        plt.ylabel('Stitch Num')
        plt.xlim(x_min, x_max)
        plt.legend()
        
        # Plot for expertise3
        plt.subplot(3, 1, 3)
        sns.histplot(data3, kde=False, bins=30, color='blue', stat='density')
        mean3, std3 = norm.fit(data3)
        x = np.linspace(x_min, x_max, 100)
        p = norm.pdf(x, mean3, std3)
        plt.plot(x, p, 'k', linewidth=2)
        plt.axvline(mean3, color='black', linestyle='--', label=f'Mean: {mean3:.4f}')
        plt.axvline(mean3 + std3, color='black', linestyle='--', label=f'STD: {std3:.4f}')
        plt.axvline(mean3 - std3, color='black', linestyle='--')
        plt.title(f'{title} - {expertise3}')
        #plt.xlabel(xlabel)
        plt.ylabel('Stitch Num')
        plt.xlim(x_min, x_max)
        plt.legend()
        plt.savefig(f'{idx}.png')
    
        #plt.tight_layout(rect=[50, 50, 50, 50])
    plt.show()
    

# Function to calculate means and stds for each metric and expertise group
def calculate_means_stds(ali_metrics, df_ali_scores):
    subjects = df_ali_scores['Subject'].tolist()
    expertise = df_ali_scores['Expertise'].tolist()
    
    metrics = [
        ('ALI_Error_4_Normalized_Lengths', 'Stitch Lengths (ALI Error 4)'),
        ('ALI_Error_6_Projections', 'Stitch Distance Along (ALI Error 6 Projections)'),
        ('ALI_Error_6_Vertical_Distances', 'Stitch Distance Vertical (ALI Error 6 Vertical Distances)'),
        ('ALI_Error_6_Stitch_Length_Projections', 'Stitch Length Projections (ALI Error 6 Stitch Length Projections)')
    ]
    
    data = {
        'Metric': [],
        'Expertise': [],
        'Mean': [],
        'Std': []
    }
    
    for metric, title in metrics:
        for exp in set(expertise):
            exp_subjects = [subj for subj, e in zip(subjects, expertise) if e == exp]
            values = [length for subj in exp_subjects for length in ali_metrics[metric].get(subj, [])]
            
            if values:
                mean = np.mean(values)
                std = np.std(values)
                data['Metric'].append(title)
                data['Expertise'].append(exp)
                data['Mean'].append(mean)
                data['Std'].append(std)
    
    return pd.DataFrame(data)

# Plot the means and stds with error bars
def plot_means_stds_with_error_bars(ali_metrics, df_ali_scores):
    df_means_stds = calculate_means_stds(ali_metrics, df_ali_scores)
    print(df_means_stds)
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Define colors for each expertise group
    colors = {'Student': 'red', 'Resident': 'green', 'Surgeon': 'blue'}
    
    # Iterate through each metric to plot
    for metric in df_means_stds['Metric'].unique():
        metric_data = df_means_stds[df_means_stds['Metric'] == metric]
        for exp in metric_data['Expertise'].unique():
            exp_data = metric_data[metric_data['Expertise'] == exp]
            plt.errorbar(exp_data['Metric'], exp_data['Mean'], yerr=exp_data['Std'], fmt='o', label=f'{exp} ({metric})', color=colors[exp], capsize=5)
    
    plt.title('Means and Standard Deviations for Each Metric by Expertise')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Expertise and Metric', loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def resize_image_to_target_longest_side(original_image, target_longest_side):
    original_width, original_height = original_image.size
    max_dimension = max(original_width, original_height)
    scale_factor = target_longest_side / max_dimension
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def paint_ground_truth_mask(image, annotations):
    mask = Image.new('L', (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)

    # Convert PIL image to OpenCV format
    image_np = np.array(image)

    for annotation in annotations:
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        for segment in segmentation:
            polygon = [tuple(segment[i:i + 2]) for i in range(0, len(segment), 2)]
            draw.polygon(polygon, outline=1, fill=1)

            # Compute the area of the polygon
            contour = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(contour)

            # Draw contours on the image in red color
            cv2.drawContours(image_np, [contour], -1, (0, 255, 0), 1)

        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(image_np, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)

    # Convert back to PIL Image
    image_with_contours = Image.fromarray(image_np)

    return image_with_contours


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union


def compute_bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def compute_distance_and_size_metrics(boxA, boxB):
    centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
    centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)

    distance = np.sqrt((centerA[0] - centerB[0]) ** 2 + (centerA[1] - centerB[1]) ** 2)

    sizeA = (boxA[2] - boxA[0], boxA[3] - boxA[1])
    sizeB = (boxB[2] - boxB[0], boxB[3] - boxB[1])

    return distance, sizeA, sizeB


def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    detected = []

    for gt_box in gt_boxes:
        matched = False
        for pred_box in pred_boxes:
            iou = compute_bbox_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                if not any((pred_box == x).all() for x in detected):
                    true_positives += 1
                    detected.append(pred_box)
                    matched = True
                    break
        if not matched:
            false_negatives += 1

    false_positives = len(pred_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def extract_loss(filename):
    # Regular expression to capture the loss
    match = re.search(r'train_loss=(\d*\.?\d+)', filename)
    return float(match.group(1)) if match else float('inf')  # Return a large number if no match

def check_serialization_issue(metrics):
    for img_name, img_metrics in metrics.items():
        try:
            json.dumps(img_metrics)
        except TypeError as e:
            print(f"Serialization issue with image: {img_name}")
            print(e)
            for key, value in img_metrics.items():
                try:
                    json.dumps(value)
                except TypeError as e:
                    print(f"Serialization issue with key: {key} in image: {img_name}")
                    print(e)
                    if isinstance(value, list):
                        for idx, item in enumerate(value):
                            try:
                                json.dumps(item)
                            except TypeError as e:
                                print(f"Serialization issue with item at index {idx} in key: {key} for image: {img_name}")
                                print(e)

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj