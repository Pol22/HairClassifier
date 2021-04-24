import dlib
import os
import cv2
import numpy as np


# dir_path = 'data256x256/data256x256_longhair'
# out_dir_path = 'transformed_data/longhair'
dir_path = 'data256x256/data256x256_shorthair'
out_dir_path = 'transformed_data/shorthair'


imgs = os.listdir(dir_path)

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# face 156x156 in mean

for img_n, img_name in enumerate(imgs):
    img_path = os.path.join(dir_path, img_name)
    if os.path.isfile(img_path):
        img = dlib.load_rgb_image(img_path)
        faces = face_detector(img, 1)
        landmark_tuple = []
        if len(faces) == 0:
            continue

        face = faces[0]
        landmarks = landmark_detector(img, face)
        for n in range(0, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
        #     cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

        routes = []

        for i in range(15, -1, -1):
            from_coordinate = landmark_tuple[i+1]
            to_coordinate = landmark_tuple[i]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[0]
        to_coordinate = landmark_tuple[17]
        routes.append(from_coordinate)

        for i in range(17, 20):
            from_coordinate = landmark_tuple[i]
            to_coordinate = landmark_tuple[i+1]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[19]
        to_coordinate = landmark_tuple[24]
        routes.append(from_coordinate)

        for i in range(24, 26):
            from_coordinate = landmark_tuple[i]
            to_coordinate = landmark_tuple[i+1]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[26]
        to_coordinate = landmark_tuple[16]
        routes.append(from_coordinate)
        routes.append(to_coordinate)

        # for i in range(0, len(routes)-1):
        #     from_coordinate = routes[i]
        #     to_coordinate = routes[i+1]
        #     img = cv2.line(img, from_coordinate, to_coordinate, (255, 255, 0), 1)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(np.bool)
        mask = np.logical_not(mask)
        
        out = np.zeros_like(img)
        out[mask] = img[mask]

        # cv2.imshow('Frame', out)
        out_path = os.path.join(out_dir_path, img_name)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_path, out)

        if img_n % 100 == 0 or img_n == len(imgs)-1:
            print(f'Part {img_n}/{len(imgs)}')
            # break

        # Press Q on keyboard to  exit
        # cv2.waitKey()

    # break