# HairClassifier

### Repository structure
- `data_transform.py` - script for dataset transformation (remove face from image)
- `model_30.pt` - trained model for hair classification (architecture located on `train.py`)
- `predict.py` - script for hair class prediction by folder with images
- `train.py` - script for model training on transformed dataset

### Usage
To use it is necessary download a trained facial shape predictor from dlib library from the [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (It's used in Python API of dlib)

__Command to generate results file__ *(Python usage)*
```bash
python predict.py --imgs_dir path/to/dir/with/images --shape_predictor shape_predictor_68_face_landmarks.dat --cls_model model_30.pt --result_file 
```
__Bash script usage__  *(with shape predictor downloading)*
```bash
./run.sh path/to/dir/with/images
```

### Improvements ideas
- Use external data on training dataset
- Better hair segmentation on input images
- Try other model architectures
- More accurate face detection and cropping from image
