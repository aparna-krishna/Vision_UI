from imgaug import augmenters as iaa
from random import randint
import operator
import numpy as np
import os
from pathlib import PurePosixPath
from PIL import Image
from fastai.vision.data import verify_images


def check_images(path):
    return verify_images(path)


def aug_size(path, basewidth):
    img_num_folder = dict()
    for folder in path:
        img_num_folder[folder] = len(os.listdir(folder))

    max_images_label = max(img_num_folder.items(),
                           key=operator.itemgetter(1))[0]
    max_images_number = img_num_folder[max_images_label]

    # Iterating thorugh every folder
    for folder in path:
        # Making sure we dont augment images for the folder that has highest number of images
        if folder == max_images_label:
            continue
        #print('\n')
        #print('Starting Augmentation for {}'.format(folder))
        # We choose a value between 5% +- of highest dataset
        images_to_be_augmented = randint(
            int(.95*max_images_number), int(1.05*max_images_number))
        # Listing the number of images
        items_in_folder = os.listdir(folder)
        #print('Current dataset: {} in {}'.format(len(items_in_folder), folder))
        # The loop runs till the len of folder doesnt reach the number we desired from above line
        while len(os.listdir(folder)) < images_to_be_augmented:
            # Chosing a random image from folder
            if not items_in_folder:
                break
            image_to_be_augmented = np.random.choice(items_in_folder)
            # Size Augmentation
            try:
                if not image_to_be_augmented.startswith('.'):
                    img = Image.open(os.path.join(folder, image_to_be_augmented))
                    for value in basewidth:
                        wpercent = (value/float(img.size[0]))
                        hsize = int((float(img.size[1])*float(wpercent)))
                        img = img.resize((value, hsize), Image.ANTIALIAS)
                        if image_to_be_augmented.endswith('.jpeg'):
                            img.save(os.path.join(folder, image_to_be_augmented.split(
                                '.jpeg')[0]+'_'+str(value)+'.jpeg'))
                        elif image_to_be_augmented.endswith('.jpg'):
                            img.save(os.path.join(folder, image_to_be_augmented.split(
                                '.jpg')[0]+'_'+str(value)+'.jpg'))
                        elif image_to_be_augmented.endswith('.png'):
                            img.save(os.path.join(folder, image_to_be_augmented.split(
                                '.png')[0]+'_'+str(value)+'.png'))
                        elif image_to_be_augmented.endswith('.tif'):
                            img.save(os.path.join(folder, image_to_be_augmented.split(
                                '.tif')[0]+'_'+str(value)+'.tif'))
            except:
                print('{} couldnot be augmented'.format(image_to_be_augmented))
                #os.remove(os.path.join(folder, image_to_be_augmented))
            # Removing the image from folder so we dont augment it again by mistake
            items_in_folder.remove(image_to_be_augmented)


def aug_affine(path, augmentors):
    for folder in path:
        #print(folder)
        for img in os.listdir(folder):
            if not img.startswith('.'):
                try:
                    img_arr = np.array(Image.open(os.path.join(folder, img)))
                    ###### Creating 1st augmented Image ######
                    augmentor = np.random.choice(augmentors)
                    example = augmentor.augment_image(img_arr)
                    example = Image.fromarray(example, 'RGB')

                    if img.endswith('.png'):
                        filename = img.split(
                            '.png')[0] + '_' + type(augmentor).__name__+'.png'
                        example.save(os.path.join(folder, filename))

                    elif img.endswith('.jpg'):
                        filename = img.split(
                            '.jpg')[0] + '_' + type(augmentor).__name__+'.jpg'
                        example.save(os.path.join(folder, filename))

                    elif img.endswith('.jpeg'):
                        filename = img.split('.jpeg')[
                            0] + '_' + type(augmentor).__name__+'.jpeg'
                        example.save(os.path.join(folder, filename))

                except Exception as e:
                    print('The image {} cannot be affine augmented'.format(img))
                    print(e)
                    #os.remove(os.path.join(folder , img))
                    continue

                ###### Creating 2nd augmented Image #####
                try:
                    augmentor = np.random.choice(augmentors)
                    example = augmentor.augment_image(img_arr)
                    example = Image.fromarray(example, 'RGB')

                    if img.endswith('.png'):
                        filename = img.split(
                            '.png')[0] + '_' + type(augmentor).__name__+'.png'
                        example.save(os.path.join(folder, filename))

                    elif img.endswith('.jpg'):
                        filename = img.split(
                            '.jpg')[0] + '_' + type(augmentor).__name__+'.jpg'
                        example.save(os.path.join(folder, filename))

                    elif img.endswith('.jpeg'):
                        filename = img.split('.jpeg')[
                            0] + '_' + type(augmentor).__name__+'.jpeg'
                        example.save(os.path.join(folder, filename))

                except Exception as e:
                    print('The image {} cannot be affine augmented'.format(img))
                    print(e)
                    #os.remove(os.path.join(folder , img))
                    continue


def augment(attribute_path, output_pp4):
    # Defining Paths to train and valid folders

    #attribute_path = '/Users/aparna/Desktop/data_tagGen/test_for_gui2'

    root_path = attribute_path#'/content/drive/My Drive/womens-dresses/silhouette'#PurePosixPath(os.getcwd())


    train_path = [
        root_path+'/train/{}'.format(attr) for attr in os.listdir(os.path.join(root_path, 'train'))]
    valid_path = [
        root_path+'/valid/{}'.format(attr) for attr in os.listdir(os.path.join(root_path, 'valid'))]
    
    # remove hidden files
    train_path = [x for x in train_path if not os.path.basename(x).startswith('.')]
    valid_path = [x for x in valid_path if not os.path.basename(x).startswith('.')]
 
    #print(train_path)
    #print(valid_path)

    '''
    # Verifying Image
    with output_pp4:
        print('Checking for corrupted images......')
        for item in train_path:
            check_images(item)

        for item in valid_path:
            #print(item)
            check_images(item)
            '''
    #print('\n')

    # Augmentors
    flipper = iaa.Fliplr(p=1.0)
    blurer = iaa.GaussianBlur(sigma=0.85)
    hue_sat = iaa.AddToHueAndSaturation(value=30)
    contrast = iaa.SigmoidContrast(cutoff=0.25)
    edge = iaa.EdgeDetect(alpha=0.5)
    crop_pad = iaa.Pad(px=(8, 0, 0, 32))
    affine = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={
                        "x": (-0.2, 0.2), "y": (-0.2, 0.2)}, shear=(-16, 16), order=[0, 1], cval=(0, 255))
    augmentors = [flipper, blurer, hue_sat, contrast, edge, crop_pad, affine]
    # Rebalance dataset
    basewidth = [800]

    aug_size(train_path, basewidth)
    aug_size(valid_path, basewidth)
    with output_pp4:
        print('\n ...Pre-Augmentation ...\n')
        print('\t in Train folders')
        for folder in train_path:
            print('Images in {} are {}'.format(folder, len(os.listdir(folder))))

        print('\n')
        print('\t in Valid folders')
        for folder in valid_path:
            print('Images in {} are {}'.format(folder, len(os.listdir(folder))))

        print('\n ...Post Augmentation... \n')
        aug_affine(train_path, augmentors)

        aug_affine(valid_path, augmentors)

        print('\t final Train folders')
        for folder in train_path:
            print('Images in {} are {}'.format(folder, len(os.listdir(folder))))

        print('\n')
        print('\t final Valid folders')
        for folder in valid_path:
            print('Images in {} are {}'.format(folder, len(os.listdir(folder))))


   
    