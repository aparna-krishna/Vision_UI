from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets
import ipywidgets as widgets
from IPython.display import display,clear_output

import pandas as pd

from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import*
from fastai.widgets import ClassConfusion

import matplotlib.pyplot as plt
import auxillary_files.webp_to_jpg_v1 as webp_to_jpg_v1
import auxillary_files.split_data as split_data
import auxillary_files.get_test_accuracy as get_test_accuracy
import auxillary_files.check_image_sanity as check_image_sanity
import auxillary_files.Data_Aug_v2 as Data_Aug_v2
from auxillary_files.radam import RAdam
from dashboards import preprocess_dashboard, metrics_dashboard, testResult_dashboard
from dashboards.architecture_dashboard import architecture_dashboard

import fastai
import psutil

import webbrowser
from IPython.display import YouTubeVideo
from ipywidgets import HBox, Label

import warnings
warnings.filterwarnings('ignore')


#widget layouts
layout = {'width':'90%', 'height': '50px', 'border': 'solid', 'fontcolor':'lightgreen'}
layout_two = {'width':'100px', 'height': '200px', 'border': 'solid', 'fontcolor':'lightgreen'}
style_green = {'handle_color': 'green', 'readout_color': 'red', 'slider_color': 'blue'}
style_blue = {'handle_color': 'blue', 'readout_color': 'red', 'slider_color': 'blue'}

########################################

## Training Dashboard code begins below ##

#########################################

def get_data():
    """ 
    this function previously called data from folder/csv/ etc based on user specification, we only use 'in_folder'
    """
    Data_in.in_folder()
    

class Data_in():
    def in_folder():
        print('\n>> In Folder')
        
        path = preprocess_dashboard.ds.data_path.value

        batch_val = int(architecture_dashboard.f.value) # batch size
        image_val = int(architecture_dashboard.m.value) # image size

        r = architecture_dashboard.pretrain_check.value

        #values for saving model
        value_mone = str(architecture_dashboard.archi.value)
        value_mtwo = str(architecture_dashboard.pretrain_check.value)
        value_mthree = str(round(architecture_dashboard.f.value))
        value_mfour = str(round(architecture_dashboard.m.value))

        train_choice = 'train'
        valid_choice = 'valid'

        Data_in.in_folder.from_code = ''

        ##### CREATING DATABUNCH:

        tfms= get_transforms(do_flip=True, max_lighting=0.01, max_rotate=10,
                      max_zoom=1.05,p_lighting=0.,p_affine=0.1, max_warp=0.01)

        print('Creating databunch...')
        data = ImageDataBunch.from_folder(path,
                                          train=train_choice,
                                          valid=valid_choice,
                                          ds_tfms=tfms,
                                          bs=batch_val,
                                          size=image_val,
                                          valid_pct=0.2)
        print('Successfully created databunch')

        ##### TRAINING BEGINS:

        architecture_dashboard.arch_work()

        if display_ui.tab.selected_index == 3 :#Train
            print('FOLDER')
            button_LR = widgets.Button(description='LR')
            button_T = widgets.Button(description='Train')
            disp = widgets.HBox([button_LR, button_T])
            display(disp)

            out_fol = widgets.Output()
            display(out_fol)
            def on_button_clicked(b):
                with out_fol:
                    clear_output()
                    a, b = metrics_dashboard.metrics_list(mets_list, mets_list_code)
                    opt_func = partial(RAdam, betas=(0.9,0.99), eps=1e-6)
                    learn = cnn_learner(data, base_arch=architecture_dashboard.arch_work.info, pretrained=r, metrics=a, opt_func=opt_func, custom_head=None)

                    learn.lr_find()
                    learn.recorder.plot()
            button_LR.on_click(on_button_clicked)

            def on_button_clicked_2(b):
                with out_fol:
                    button = widgets.Button(description='Train_N')
                    clear_output()
                    training_ds()
                    display(button)
                    def on_button_clicked_3(b):
                        lr_work()
                        a, b = metrics_dashboard.metrics_list(mets_list, mets_list_code)
                        b_ = b[0]
                        learn = cnn_learner(data, base_arch=architecture_dashboard.arch_work.info, pretrained=r, metrics=a, custom_head=None)
                        print(f'Training in folder......{b}')
                        cycle_l = int(training_ds.cl.value)

                        #save model
                        file_model_name = value_mone + '_pretrained_' + value_mtwo + '_batch_' + value_mthree + '_image_' + value_mfour

                        accuracy_out = widgets.Output()
                        display(accuracy_out)
                        with accuracy_out:
                            learn.fit_one_cycle(cycle_l,
                                                slice(lr_work.info),
                                                callbacks=[SaveModelCallback(learn, every='improvement', monitor=b_, name='best_'+ file_model_name)])
                        learn.export(file_model_name)

                    button.on_click(on_button_clicked_3)
            button_T.on_click(on_button_clicked_2)


###########################
## Modules for train tab ##
###########################
def stats_info():

    if architecture_dashboard.norma.value == 'Imagenet':
        stats_info.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    stats = stats_info.stats

def info_lr():

    batch_val = int(architecture_dashboard.f.value) # batch size
    image_val = int(architecture_dashboard.m.value) # image size
    
    button = widgets.Button(description='Review Parameters')
    button_two = widgets.Button(description='LR')
    button_three = widgets.Button(description='Train')

    butlr = widgets.HBox([button, button_two, button_three])
    display(butlr)

    out = widgets.Output()
    display(out)

    def on_button_clicked_info(b):
        with out:
            clear_output()

            print(f'Data in: {preprocess_dashboard.ds.data_path.value}| Normalization: {architecture_dashboard.norma.value}| Architecture: {architecture_dashboard.archi.value}| Pretrain: {architecture_dashboard.pretrain_check.value}|'
            f'Batch Size: {architecture_dashboard.f.value}| Image Size: {architecture_dashboard.m.value}')

            print(f'Training Metrics: {metrics_dashboard.metrics_list(mets_list)} ')

    button.on_click(on_button_clicked_info)

    def on_button_clicked_info2(b):
        with out:
            clear_output()
            learn_dash()

    button_two.on_click(on_button_clicked_info2)

    def on_button_clicked_info3(b):
        with out:
            clear_output()
            print('Train')
            training()

    button_three.on_click(on_button_clicked_info3)

def lr_work():
    if training_ds.lr.value == '1e-6':
        lr_work.info = float(0.000001)
    elif training_ds.lr.value == '1e-5':
        lr_work.info = float(0.00001)
    elif training_ds.lr.value == '1e-4':
        lr_work.info = float(0.0001)
    elif training_ds.lr.value == '1e-3':
        lr_work.info = float(0.001)
    elif training_ds.lr.value == '1e-2':
        lr_work.info = float(0.01)
    elif training_ds.lr.value == '1e-1':
        lr_work.info = float(0.1)

def training_ds():
    print(">> Using fit_one_cycle \n >> Model saved as ('architecture' + 'pretrained' + batchsize + image size) in model path")
    print(">> Best model also saved as (best_'architecture' + 'pretrained' + batchsize + image size)")
    button = widgets.Button(description='Train')

    style = {'description_width': 'initial'}

    training_ds.cl=widgets.FloatSlider(min=1,max=64,step=1,value=1, continuous_update=False, layout=layout, style=style_green, description="Cycle Length")
    training_ds.lr = widgets.ToggleButtons(
        options=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1'],
        description='Learning Rate:',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        style=style,
        value='1e-2',
        tooltips=['Choose a suitable learning rate'],
    )

    display(training_ds.cl, training_ds.lr)

def learn_dash():
    button = widgets.Button(description="Learn")
    print ('Chosen metrics: ',metrics_dashboard.metrics_list(mets_list))
    metrics_dashboard.metrics_list(mets_list)

    
    batch_val = int(architecture_dashboard.f.value) # batch size
    image_val = int(architecture_dashboard.m.value) # image size

    r = architecture_dashboard.pretrain_check.value
    t = metrics_dashboard.metrics_list(mets_list)

    path = preprocess_dashboard.ds.data_path

    tfms= get_transforms(do_flip=True, max_lighting=0.01, max_rotate=10,
                      max_zoom=1.05,p_lighting=0.,p_affine=0.1, max_warp=0.01)
    
    # assumes split valid etc all created, needs a test file
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    learn = cnn_learner(data, base_arch=architecture_dashboard.arch_work.info, pretrained=r, metrics=metrics_dashboard.metrics_list(mets_list,mets_list_code), custom_head=None)

    lr_out = widgets.Output()
    display(lr_out)

    with lr_out:
        learn.lr_find()     
        learn.recorder.plot()