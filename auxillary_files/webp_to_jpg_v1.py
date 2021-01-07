from PIL import Image
import os
from pathlib import PurePosixPath
import shutil
import ipywidgets as widgets


# images_path = os.path.join(path, 'Images')
# images = os.listdir(images_path)
# if not os.path.exists("Converted_images"):
#     os.mkdir('Converted_images')
# converted_images_path = os.path.join(path, 'Converted_images')

# for filename in images:
#     if filename.endswith('.webp'):
#         print('found: ' + os.path.splitext(filename)[0])
#         print('converting to: ' + os.path.splitext(filename)[0] + '.jpg')
#         im = Image.open(os.path.join(images_path, filename)).convert("RGB")
#         im.save(os.path.splitext(filename)[0] + '.jpg', "jpeg")
#         shutil.move(os.path.splitext(filename)[
#                     0] + '.jpg', os.path.join(converted_images_path, os.path.splitext(filename)[0] + '.jpg'))
#         print('done converting…')

'''
if __name__ == '__main__':
    # Creating paths
    root_path = PurePosixPath(os.getcwd())
    train_path = root_path/'train'

    for folder in os.listdir(train_path):
        image_list = os.listdir(os.path.join(train_path, folder))
        for filename in image_list:
            if filename.endswith('.webp'):
                im = Image.open(os.path.join(
                    train_path, folder, filename)).convert("RGB")
                im.save(os.path.join(train_path, folder,
                                     os.path.splitext(filename)[0] + '.jpg'), "jpeg")
                os.remove(os.path.join(train_path, folder, filename))

    print('done converting…')
    '''

def convert(path):
    root_path = PurePosixPath(path)
    train_path = root_path/'train'

    for folder in os.listdir(train_path):
        if not folder.startswith('.'): 
            image_list = os.listdir(os.path.join(train_path, folder))
            for filename in image_list:
                if filename.endswith('.webp'):
                    im = Image.open(os.path.join(
                        train_path, folder, filename)).convert("RGB")
                    im.save(os.path.join(train_path, folder,
                                        os.path.splitext(filename)[0] + '.jpg'), "jpeg")
                    os.remove(os.path.join(train_path, folder, filename))

   