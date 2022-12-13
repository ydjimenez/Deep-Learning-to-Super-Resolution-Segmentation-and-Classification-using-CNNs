from skimage.measure import compare_ssim
import argparse
import glob
import cv2

# 2. Construya el análisis de argumentos y analice los argumentos
ap = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser1 = argparse.ArgumentParser()
parser.add_argument("-f", "--first", required=True, help="Directorio de la imagen que se comparará")
parser.add_argument("-s", "--second", required=True, help="Directorio de la imagen que se utilizará para comparar")
#args = vars(ap.parse_args())
args = parser.parse_args()
veces=0

# 3. Cargue las dos imágenes de entrada
archivo4 = open('estadisticas/smmi.txt', 'w')
image_list = sorted(glob.glob('{}/*'.format(args.first)))
image_list1 = sorted(glob.glob('{}/*'.format(args.second)))


for i, image_path in enumerate(image_list):
	print('el phat uno ',image_path)
	imageA = cv2.imread(image_path)
	cadena = image_path[13:]
	print(cadena)

	for e, image_path1 in enumerate(image_list1):
		cadena2 = image_path1[7:]

		print("la cadena 1",cadena)
		print("la cadena 2",cadena2)




		#print('el phat dos ',image_path1)
		if  cadena==cadena2:
			veces=veces+1
			print("son iguales ")
			print("direccion 1 es :  ",image_path)
			imageB = cv2.imread(image_path1)
			print("direccion 2 es :  ",image_path1)
			grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
			grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
			(score, diff) = compare_ssim(grayA, grayB, full=True)
			diff = (diff * 255).astype("uint8")
			print(veces,"SSIM: {}".format(round(score,2)))
			parseo = str(veces)
			archivo4.write(parseo)
			archivo4.write(' ,'+cadena)

			archivo4.write("  , {}\n".format(round(score,2)))

		pass


	print("nueva imagen fin for1 ")
print("Total de datos ",veces)
archivo4.close()
	# 4. Convierte las imágenes a escala de grises
	#grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	#grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	# 5. Calcula el índice de similitud estructural (SSIM) entre los dos
	# imágenes, asegurándose de que se devuelva la imagen de diferencia
	#(score, diff) = compare_ssim(grayA, grayB, full=True)
	#diff = (diff * 255).astype("uint8")
	# 6. Puede imprimir solo el record si lo desea
	#print("SSIM: {}".format(round(score,2)))