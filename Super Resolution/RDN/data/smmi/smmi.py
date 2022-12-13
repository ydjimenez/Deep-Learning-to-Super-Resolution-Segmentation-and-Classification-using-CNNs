from skimage.measure import compare_ssim
import argparse
import glob
import cv2

# 2. Construya el análisis de argumentos y analice los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Directorio de la imagen que se comparará")
ap.add_argument("-s", "--second", required=True, help="Directorio de la imagen que se utilizará para comparar")
args = vars(ap.parse_args())

# 3. Cargue las dos imágenes de entrada

image_list = sorted(glob.glob('{}/*'.format(ap.add_argumen)))
image_list1 = sorted(glob.glob('{}/*'.format(ap.add_argumen)))
for i, image_path in enumerate(image_list):
	for i, image_path1 in enumerate(image_list):
		imageA = cv2.imread(image_path)
		imageB = cv2.imread(image_path1 )

		print("el for funciona ")
	# 4. Convierte las imágenes a escala de grises
	#grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	#grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	# 5. Calcula el índice de similitud estructural (SSIM) entre los dos
	# imágenes, asegurándose de que se devuelva la imagen de diferencia
	#(score, diff) = compare_ssim(grayA, grayB, full=True)
	#diff = (diff * 255).astype("uint8")
	# 6. Puede imprimir solo el record si lo desea
	#print("SSIM: {}".format(score))