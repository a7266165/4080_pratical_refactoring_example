import cv2
import mediapipe as mp
import numpy as np
import os


FACEMESH_MID_LINE = frozenset([(10, 151), (151, 9), (9, 8), (8, 168), 
                               (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                                (4, 1), (1, 19), (19, 94), (94, 2)
                                ])

def mid_line_angle_all_points(results, height, width):
    angles = []
    for pair in FACEMESH_MID_LINE:
        point1 = results.multi_face_landmarks[0].landmark[pair[0]]
        point2 = results.multi_face_landmarks[0].landmark[pair[1]]
        dot1 = np.array([point1.x * width, point1.y * height, 0])
        dot2 = np.array([point2.x * width, point2.y * height, 0])

        vector1 = dot2 - dot1
        if vector1[1] == 0:
            angle1_deg = 90.0
        else:
            angle1 = np.arctan(vector1[0] / vector1[1])
            angle1_deg = np.degrees(angle1)

        angles.append(angle1_deg)
        avg_angle = sum(angles) / len(angles)

    return avg_angle

# 因為路徑寫法不同，做第二版
def rotate_image(save_dir, image_folder, face_mesh):
    print('Processing', image_folder)
    # 取出子資料夾名稱
    subname = os.path.basename(image_folder)
    target_folder = os.path.join(save_dir, subname)
    os.makedirs(target_folder, exist_ok=True)

    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith('.jpg'):
            continue
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        angle = mid_line_angle_all_points(results, h, w)
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(os.path.join(target_folder, img_name), rotated)

# 遍歷parent_dir的每個子目錄
if __name__ == '__main__':

    try:

        parent_dir = r'c:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic_temp'
        save_dir = r'c:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        for child_dir in os.listdir(parent_dir):
            child_path = os.path.join(parent_dir, child_dir)
            rotate_image(save_dir, child_path, face_mesh)

        print('Done')
        face_mesh.close()

    except Exception as e:
        print('Error:', e)