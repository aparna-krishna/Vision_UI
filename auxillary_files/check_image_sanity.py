import concurrent.futures
import PIL.Image
import cv2
import os

def check_image_sanity(img_pth):
    """
    Given an image path, loads the image using PIL and cv2 and looks at the sizes of the images.
    If they're not loading properly, then the function returns false else true
    """
    sane = True
    try:
        # Load the image using PIL and cv2
        PImg = PIL.Image.open(img_pth)
        cvImg = cv2.imread(img_pth)

        # Double check the size of image and the number of channels of the image
        w, h = PImg.size
        cv_h, cv_w, cv_chn = cvImg.shape

        height_check = (w == cv_w)
        width_check = (h == cv_h)
        channel_check = (cv_chn == 3)

        sane = height_check & width_check & channel_check
        
    except Exception as e:
        sane = False
    
    return (img_pth, sane)

# Get the paths to all the images in train and validation folder and train
def clean_images(base_path):
    all_files = []
    #base_path = "/home/vinayak/Desktop/womens-dress-collar/womens_dress_collar_modelled_no_duplicates"
    train = base_path+'/train'
    valid = base_path+'/valid'
    for root, dirs, files in os.walk(train):
        for name in files:
            all_files.append(os.path.join(root, name))

    for root, dirs, files in os.walk(valid):
        for name in files:
            all_files.append(os.path.join(root, name))

    # Validate the sanity of images
    with concurrent.futures.ProcessPoolExecutor() as e:
        result = e.map(check_image_sanity, all_files)

    # Get all the faulty images from the result generator above
    faulty_images = []
    for item in result:
        if item[1] == False:
            faulty_images.append(item[0])

    # Optionally store the names of all the faulty images
    # with open("faulty_images.txt", "w") as f:
    #     for item in faulty_images:
    #         f.writelines(f"{item}\n")
    #     f.close()

    # Remove all the faulty images
    for img in faulty_images:
        os.remove(img)

  
#clean_images("/Users/aparna/Desktop/data_tagGen/silhouette_noAug_may2020")