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

## Modules for Architecture Dashboard ##

########################################

def architecture_dashboard():

    """GUI for architecture selection as well as batch size, image size"""
 
    style = {'description_width': 'initial'}

    layout = widgets.Layout(width='auto', height='40px') #set width and height

   

    button = widgets.Button(description='Check System. (Cuda must return true to continue)',
                         layout=layout, alight_items='stretch')
    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked_info(b):
        with out:
            clear_output()
            print(f'Fastai Version: {fastai.__version__}')
            print(f'Cuda: {torch.cuda.is_available()}')
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Python version: {sys.version}')
            print(psutil.cpu_percent())
            print(psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])

    button.on_click(on_button_clicked_info)

    architecture_dashboard.norma = widgets.ToggleButtons(
        options=['Imagenet'],
        description='Normalization:  ',
        disabled=False,
        value='Imagenet',
        button_style='info', 
        tooltips=['Imagenet stats', 'Create your own', 'Cifar stats', 'Mnist stats'],
        style=style
    )
    architecture_dashboard.archi = widgets.ToggleButtons(
        options=['resnet50'],
        description='Architecture:  ',
        disabled=False,
        value='resnet50',
        button_style='info',
        tooltips=[],
    )
    layout = widgets.Layout(width='60%', height='40px') #set width and height
    architecture_dashboard.pretrain_check = widgets.Checkbox(
        options=['Yes', "No"],
        description='Pretrained:',
        disabled=True,#False, # we always want it to be pretrained
        value=True,
        box_style='success',
        button_style='lightgreen',
        tooltips=['Default: Checked = use pretrained weights, Unchecked = No pretrained weights'],
    )
    architecture_dashboard.method = widgets.ToggleButtons(
        options=['cnn_learner'],
        description='Model Method:',
        disabled=False,
        value='cnn_learner',
        button_style='info', 
        tooltips=['Under construction'],
        style=style
    )

    architecture_dashboard.f=widgets.FloatSlider(min=8,max=64,step=8,value=64, continuous_update=False, layout=layout, style=style_green)
    architecture_dashboard.m=widgets.FloatSlider(min=0, max=360, step=16, value=225, continuous_update=False, layout=layout, style=style_green)
    architecture_dashboard.bs = HBox([Label('BatchSize (default 64)'), architecture_dashboard.f])
    architecture_dashboard.imsize = HBox([Label('ImageSize (default 225)'), architecture_dashboard.m])

    display(architecture_dashboard.norma, architecture_dashboard.archi, architecture_dashboard.pretrain_check, architecture_dashboard.method, architecture_dashboard.bs, architecture_dashboard.imsize)



def arch_work():
    arch_work.info = models.resnet50