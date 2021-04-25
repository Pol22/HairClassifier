import os
import argparse
import dlib
import csv
import cv2
import torch
import numpy as np

from train import Net


class HairClassifier(object):
    def __init__(self, shape_predictor, cls_model):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor(args.shape_predictor)
        # model loading
        self.model = Net()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(cls_model, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded successfully')
        self.model.to(self.device)
        self.model.eval()
        # crop params
        self.crop_size = 256
        self.face_size = 156 # mean face size in dataset
        # pads around detected face
        self.bottom_pad = 20 # below the face contains less info about hair 
        self.top_pad = self.crop_size - self.bottom_pad
        self.left_pad = (self.crop_size - self.face_size) // 2
        self.right_pad = self.left_pad

    def crop_face(self, img_path):
        img = dlib.load_rgb_image(img_path)
        faces = self.face_detector(img, 1)
        landmark_tuple = []
        if len(faces) == 0:
            return -1

        face = faces[0]
        landmarks = self.landmark_detector(img, face)
        for n in range(0, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))

        # delete face from image accordingly landmarks
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

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(np.bool)
        mask = np.logical_not(mask)
        
        out = np.zeros_like(img)
        out[mask] = img[mask]

        # crop bigger then face
        side = max(face.width(), face.height())
        ratio = side / self.face_size
        height, width , _ = out.shape

        left = max(0, face.left() - int(self.left_pad * ratio))
        right = min(width, face.right() + int(self.right_pad * ratio))
        top = max(0, face.top() - int(self.top_pad * ratio))
        bottom = min(height, face.bottom() + int(self.bottom_pad * ratio))

        out = out[top:bottom, left:right, :]
        out = dlib.resize_image(out, self.crop_size, self.crop_size)

        return out

    def predict(self, img_path):
        crop = self.crop_face(img_path)
        if isinstance(crop, int): # return -1
            return crop

        crop = np.float32(crop)
        crop /= 255.0
        crop = np.transpose(crop, (2, 0, 1))
        crop = np.expand_dims(crop, axis=0)
        crop = torch.from_numpy(crop).to(self.device)
        pred = self.model(crop)
        pred = torch.argmax(pred, 1)

        return pred.numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', required=True, 
                        help='path to dir with images')
    parser.add_argument('--shape_predictor', required=True,
                        help='path to shape predictor from dlib')
    parser.add_argument('--cls_model', required=True,
                        help='path to classifier model')
    parser.add_argument('--result_file', default='result.csv',
                        help='file for result output')
    args = parser.parse_args()

    if not os.path.isdir(args.imgs_dir):
        raise Exception('Images dir is not right!')

    classifier = HairClassifier(args.shape_predictor, args.cls_model)

    with open(args.result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE,
                            delimiter=',', quotechar='')
        imgs_list = os.listdir(args.imgs_dir)
        for i, img_name in enumerate(imgs_list):
            img_path = os.path.join(args.imgs_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            pred_class = classifier.predict(img_path)
            # 0 - longhair, 1 - shorthair

            # in output reversed classes
            if pred_class != -1:
                pred_class = 1 - pred_class
            # 1 - longhair, 0 - shorthair

            # log str
            print(f'Step {i + 1}/{len(imgs_list)}', img_name, pred_class)

            writer.writerow([str(img_path), pred_class])
