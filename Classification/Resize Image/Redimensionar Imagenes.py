
from PIL import Image
import glob

image_list = []
resized_images = []

"""REDIMENSIONAR IMAGENES BENIGNAS"""
for filename in glob.glob('redimensionar\dataB\*.jpg'):
    print(filename)
    img = Image.open(filename)
    image_list.append(img)

for image in image_list:
    #image.show()
    image = image.resize((128,128))
    resized_images.append(image)

for(i, new) in enumerate(resized_images):
        new.save('{}{}{}'.format('redimensionar\dataBenigno\imagenB_0', i+1,'.jpg'))   
        
# """REDIMENSIONAR IMAGENES MALIGNAS"""
 
# for filename in glob.glob('redimensionar\dataM\*.jpg'):
#     print(filename)
#     img = Image.open(filename)
#     image_list.append(img)

# for image in image_list:
#     #image.show()
#     image = image.resize((128,128))
#     resized_images.append(image)

# for(i, new) in enumerate(resized_images):
#         new.save('{}{}{}'.format('redimensionar\dataMaligno\imagenM_0', i+1,'.jpg'))  