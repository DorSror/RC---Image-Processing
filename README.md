# Requires the following libraries:
- cv2 (python-opencv2, version 4.10.0)
- pyyaml (pyyaml)
- face-recognition (requires cmake, wheel, dlib)
- cmake
- wheel
- dlib
- dill

# Files:
- config.yaml - stores the settings for the image processing scripts.
- main_image_proc.py - The primary image processing script, for now implements color mask/filtration and face detection using haar cascades.
- face_recog_train.py - Uses face-recognition to train and store new faces.

# Folders:
- .venv_py3.x - For now these are backed up on the repository to show which venvs we use. Do not use these, instead configure your own.
- ImageDir - Folder used to store images for future training and computations.
- haar - stores the haar cascades in .xml format.
