import os
from fastai.vision import *
import pandas as pd
from auxillary_files.radam import RAdam

def predict_fn(images,test_path, model_path):
    output = dict()
    for item in os.listdir(test_path):
        img = open_image(os.path.join(test_path,item))
        # split model name to get below format
        learner = load_learner('/Users/aparna/Desktop/data_tagGen', '95radam_silhouette.pkl')
        pred = str(learner.predict(img)[0])
        output[item] = pred
    df = pd.DataFrame(list(output.items()), columns=['ImgName', 'Preds'])       
    return df

def predict_all(test_path, model_path, save_path):
    images = os.listdir(test_path)
    result = predict_fn(images,test_path, model_path) 
    result.to_csv(save_path)
    result.head()
    return result

def get_accuracy(test_path,model_path,save_path,true_values, attribute_name):
    model_preds = predict_all(test_path, model_path, save_path)
    true_df = pd.read_csv (true_values)
    true_df = true_df[['ImgName',attribute_name]]
    true_df['ImgName'] = true_df['ImgName'].str.strip()  # removing some excess tabs/whitespace that were in my csv file
    test_df = pd.read_csv (model_preds) 
    test_df = test_df[['ImgName','Preds']]
    true_df['ImgName'] = true_df['ImgName'].str.strip()
    #print(true_df.head())
    number_of_test_images = len(test_df.index)
    test_image_list = test_df['ImgName'].tolist()
    count = 0
    for img in test_image_list:
        test_silhouette = test_df.loc[test_df['ImgName'] == img, 'Preds'].item()
        true_silhouette = true_df.loc[true_df['ImgName'] == img, attribute_name].item()
        if(test_silhouette == true_silhouette):
            count +=1

    accuracy = (count / number_of_test_images) * 100
    return accuracy