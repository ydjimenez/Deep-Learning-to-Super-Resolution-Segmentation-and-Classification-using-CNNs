import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import glob
#img = cv2.imread('ImgPrueba128/DDSM_07.png')
#height, width = img.shape[:2]
ap = argparse.ArgumentParser()
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--first", required=True, help="Directorio de la imagen que se comparará")
args = parser.parse_args()
image_list = sorted(glob.glob('{}/*'.format(args.first)))
contador=0
for i, image_path in enumerate(image_list):
    print('el phat uno ', image_path)
    img = Image.open(image_path)
    width, height = img.size
    print(height, width)
    cadena2 = image_path[13:]
    if height== 114 :
        print("el alto si es de 128")
        if width == 126 :
            #new_img = img.resize((120, 11100))
            print("el ancho si es de 128")
            new_img.save(f'129/'+cadena2)
            contador=contador+1

print("total de imagens redimensiondas  "+str(contador))



#if height==128:
#    print("el alto si es de 128")
#    if width == 128:
#        print("el ancho si es de 128")
#        res = resize(img, (126, 126))
#        new_img = img.resize((126, 126))
#        new_img.save('Grises.png')
        #cv2.imwrite('128/Grises.png',img)

#im = plt.imread('filename.jpeg')
#res = resize(im, (140, 54))




