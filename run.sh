curl http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 --output shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
python predict.py --imgs_dir $1 --shape_predictor shape_predictor_68_face_landmarks.dat --cls_model model_30.pt
