import torch
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')


    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)



    ## 2. run inference
    model = create_model(opt)
    model.single_image_inference(img, output_path)

    print('inference {} .. finished.'.format(img_path))

if __name__ == '__main__':
    main()

