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

## Modules for PreProcessing Dashboard ##

#########################################



def ds():

    ''' 
    Given a data path, this function runs our preprocessing steps and 
    saves the updated dataset as global variable ds.data_path to be accessed later during training
    '''

    style = {'description_width': 'initial'}
    layout = layout=widgets.Layout(width='10%', height='35px')
    
    # buttons for the preprocessing steps

    ds.data_path = widgets.Textarea(placeholder='enter path',disabled=False, layout=widgets.Layout(width='50%', height='30px'))
    ds.data_btn = widgets.Button(description='Confirm', button_style='info', style=style)
    ds.jpg = widgets.Button(description='Convert webp to jpg', button_style='info', style=style)
    ds.split = widgets.Button(description='Split into train / valid', button_style='info', style=style)
    ds.clean = widgets.Button(description='Remove corrupted images', button_style='info', style=style)
    ds.augment = widgets.Button(description='Augment the data',button_style='info', style=style)


    # labels for the preporcessing steps

    label1 = Label(' >>    Enter path to your attribute data:', layout=widgets.Layout(height='40px'))
    display(label1)

    enter_path = HBox([Label('Step 0:', layout=layout),ds.data_path, ds.data_btn])
    display(enter_path)

    label2 = Label(' >>    Complete the 3 preprocessing steps:', layout=widgets.Layout(height='40px'))
    display(label2)
 
    step1 =  HBox([Label('Step 1:',layout=layout), ds.jpg])
    step2 =  HBox([Label('Step 2:',layout=layout), ds.split])
    step3 = HBox([Label('Step 3:',layout=layout), ds.clean])
    step4 = HBox([Label('Step 4:',layout=layout), ds.augment])

    # display all of the above
    display(step1,step2,step3, step4)
    label3 = Label(" >>    Current status:", layout=widgets.Layout(height='40px'))
    display(label3)

    output_pp1 = widgets.Output()
    output_pp2 = widgets.Output()
    output_pp3 = widgets.Output()
    output_pp4 = widgets.Output()
    output_pp5 = widgets.Output()
    display(output_pp1, output_pp2, output_pp3, output_pp4, output_pp5)


    # BUTTON1 FUNCTIONALITY: calls function to select dataset path (currently path_choice is hardcoded, normally a file window)
    
    def confirm_data_path(b):
        # add a check that path is valid
        data_path = ds.data_path.value 
        start = "\033[1m"
        end = "\033[0;0m"
        with output_pp1:
            print(start+'Data path: '+data_path+end)
    ds.data_btn.on_click(confirm_data_path)
    
    
    # BUTTON1 FUNCTIONALITY: calls webp_to_jpg function
    def clicked1(b):
        webp_to_jpg_v1.convert(ds.data_path.value)
        with output_pp2:
            print('Step 1 done: finished converting to jpg...')
    ds.jpg.on_click(clicked1)

     # BUTTON2 FUNCTIONALITY: calls split function
    def clicked2(b):
        # execute split.py with path_choice.path as the path variable...)
        train_path = ds.data_path.value + '/train'
        valid_path = ds.data_path.value + '/valid'
        split_data.create_validation_set(train_path, valid_path)
        with output_pp3:
            print("Step 2 done: splitting into train/valid is complete...")
    ds.split.on_click(clicked2)


    # BUTTON4 FUNCTIONALITY: clean images, remove corrupted images /gifs etc
    def clicked4(b):
        check_image_sanity.clean_images(ds.data_path.value)
        with output_pp4:
            print('Step 3 done: finished removing corrupted images...')
        
    ds.clean.on_click(clicked4)

     # BUTTON3 FUNCTIONALITY: calls augment function
    def clicked3(b):
        Data_Aug_v2.augment(ds.data_path.value, output_pp4)
        with output_pp5:
            print('Final step done: augmentation is complete!')
   
        
    ds.augment.on_click(clicked3)

    return ds.data_path

