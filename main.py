from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# import argparse
import cv2
import shutil

import os, sys
from py_hardware_binding import authenticate



# handle command line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = ap.parse_args()

model_path = './gender_classification/gender_classification.model'

def main(application_path, path_to_session):
    
    path_to_photos  = input('Please give me a path to folder with photos \n')
    valid_images = [".jpg",".gif",".png",".tga", ".jpeg"]
    
    if not os.path.exists( os.path.join(path_to_photos, 'mans/')):
        os.makedirs(os.path.join(path_to_photos, "mans/"))
    if not os.path.exists( os.path.join(path_to_photos, 'womans/')):
        os.makedirs(os.path.join(path_to_photos, "womans/"))
      
    photo_counter = 0
      
    for f in os.listdir(path_to_photos):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        image = cv2.imread(os.path.join(path_to_photos, f))

        if image is None:
            print("Could not read input image")
            exit()

        # preprocessing
        output = np.copy(image)
        image = cv2.resize(image, (96,96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load pre-trained model
        model = load_model(model_path)

        # run inference on input image
        confidence = model.predict(image)[0]
        
        if confidence[0] > .9:
            shutil.move( os.path.join(path_to_photos, f), os.path.join(path_to_photos, 'mans/', f)) 
            
        if confidence[1] > .9:
            shutil.move( os.path.join(path_to_photos, f), os.path.join(path_to_photos, 'womans/', f)) 

        photo_counter+=1
        
    print('Finded ', photo_counter, ' photos in current directory')
        




@authenticate
def run(WORKDIR):    
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(WORKDIR))

    path_to_license = os.path.join(base_path, 'cfg/license.txt')
    path_to_session = os.path.join(base_path, 'cfg/anon.session')
    
    
    # determine if the application is a frozen `.exe` (e.g. pyinstaller --onefile) 
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    # or a script file (e.g. `.py` / `.pyw`)
    elif __file__:
        application_path = WORKDIR
    
    
    main(
        application_path, 
        path_to_session
    )
    