import os.path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import glob
from skimage.measure import compare_ssim
import argparse

import cv2
import cv2

from models import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=False)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--outputpath', type=str, required=True)
    args = parser.parse_args()

    archivo3 = open('estadistica/psnr.txt', 'w')


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    state_dict = model.state_dict()

    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    veces=0
    image_list = sorted(glob.glob('{}/*'.format(args.images_dir)))
    for i, image_path in enumerate(image_list):
        # image-file.image_list.str()
        # print(image-file)
        image = pil_image.open(image_path.__str__()).convert('RGB')
        # print(image.resize)
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        # print(image_width)
        # print(image_height)
        #
        hr = image.resize((image_width, image_height))
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale))
        # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # #bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)
        with torch.no_grad():
            preds = model(lr).squeeze(0)

        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

        preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

        psnr = calc_psnr(hr_y, preds_y)
        veces=veces+1

        print('PSNR: {:.2f}'.format(psnr))
        parseo = str(veces)
        archivo3.write(parseo)


        output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        # output.save(args.image_file.replace('.', '_rdn_x{}.'.format(args.scale)))

        cadena = image_path[13:]
        print(cadena)
        archivo3.write(' ,'+cadena)
        archivo3.write('   ,{:.2f}\n'.format(psnr))

        output.save(os.path.join(args.outputpath, cadena))

        # sewar.full

    #sewar.full_ref.ssim(image_path.__str__(), output, ws=11, K1=0.01, K2=0.03, MAX=Ninguno, fltr_specs=Ninguno,modo='válido')

    # imageA = cv2.imread(args.image_file)
    # imageB = cv2.imread(args.image_file.replace('.', '_rdn_x{}.'.format(args.scale)))

    # 4. Convierte las imágenes a escala de grises
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 5. Calcula el índice de similitud estructural (SSIM) entre los dos
    # imágenes, asegurándose de que se devuelva la imagen de diferencia
    # (score, diff) = compare_ssim(grayA, grayB, full=True)
    # diff = (diff * 255.0).astype("uint8")

    # 6. Puede imprimir solo el record si lo desea
    # print("SSIM: {}".format(score))
    #archivo4.write("SSIM: {}".format(score))

    #sewar.full

    archivo3.close()

