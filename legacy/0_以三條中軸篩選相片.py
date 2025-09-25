'''
以三條中軸計算臉部的角度，選擇角度最小的十張相片
'''

import cv2
import mediapipe as mp
import numpy as np
import os

def mid_line_angle_4_points(results, height, width):
    point1 = results.multi_face_landmarks[0].landmark[10]
    point2 = results.multi_face_landmarks[0].landmark[168]
    point3 = results.multi_face_landmarks[0].landmark[4]
    point4 = results.multi_face_landmarks[0].landmark[2]
    dot1 = np.array([point1.x * width , point1.y * height, 0])
    dot2 = np.array([point2.x * width , point2.y * height, 0])
    dot3 = np.array([point3.x * width, point3.y * height, 0])
    dot4 = np.array([point4.x * width, point4.y * height, 0])
    vector1 = dot2 - dot1
    vector2 = dot3 - dot2
    vector3 = dot4 - dot3
    dot_product1 = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle1 = np.arccos(dot_product1 / norm1)
    dot_product2 = np.dot(vector2, vector3)
    norm2 = np.linalg.norm(vector2) * np.linalg.norm(vector3)
    angle2 = np.arccos(dot_product2 / norm2)
    angle1_deg = np.degrees(angle1)
    angle2_deg = np.degrees(angle2)
    return angle1_deg + angle2_deg

def select_picture(image_folder, save_dir, face_mesh):
    angles = {}
    for image_name in os.listdir(image_folder):
        if not image_name.endswith('.jpg'):
            continue
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            continue
        
        total_angle = mid_line_angle_4_points(results, height, width)
        angles[image_name] = total_angle
    
    sorted_images = sorted(angles.items(), key=lambda item: item[1])[:10]
    selected_pictures = [img[0] for img in sorted_images]

    for pic in selected_pictures:
        src_path = os.path.join(image_folder, pic)
        dst_path = os.path.join(save_dir, pic)
        image = cv2.imread(src_path)
        cv2.imwrite(dst_path, image)

# 遍歷parent_dir的每個子目錄
if __name__ == '__main__':

    parent_dir = r"c:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic"
    save_root = r'c:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic_temp'

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8)

    for child_dir in os.listdir(parent_dir):
        child_path = os.path.join(parent_dir, child_dir)
        save_dir = os.path.join(save_root, child_dir)
        os.makedirs(save_dir, exist_ok=True)

        print('Processing', child_path)
        select_picture(child_path, save_dir, face_mesh)

    print('Done')
    face_mesh.close()