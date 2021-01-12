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
from dashboards import preprocess_dashboard
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



##########################
## Code for metrics tab ##
##########################

mets_list = []
mets_list_code = []

def metrics_dashboard():
    """Metrics dashboard"""
    button = widgets.Button(description="Metrics", button_style='info')

    metrics_dashboard.accuracy = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Accuracy:',
        value='Yes',
        disabled=False,
        button_style='success', 
        tooltips=[''],
    )
 
    metrics_dashboard.recall = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Recall:',
        value='No',
        disabled=False,
        button_style='success',
        tooltips=[''],
    )
    metrics_dashboard.precision = widgets.ToggleButtons(
        options=['Yes', 'No'],
        description='Precision:',
        value='No',
        disabled=False,
        button_style='success', 
        tooltips=[''],
    )
  
    layout = widgets.Layout(width='auto', height='40px') 

    ui = widgets.HBox([metrics_dashboard.accuracy,metrics_dashboard.recall, metrics_dashboard.precision])
    ui3 = widgets.VBox([ui])

    r = architecture_dashboard.pretrain_check.value

    display(ui3)

    
    metrics2 = HBox([Label("Click to view chosen metrics:",layout=widgets.Layout(width='20%', height='30px')),button])
    print()
    print()
    display(metrics2)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            a, b = metrics_list(mets_list, mets_list_code)
            print('Training Metrics''\n')
            print(a, b)
    button.on_click(on_button_clicked)

def metrics_list(mets_list, mets_list_code):
    """
    Helper to get metrics based on the user specifications 
    (TODO: remove the extra ones we are not using, currently kept it in case we wish to include it later)
    """
    mets_error = None
    mets_accuracy= metrics_dashboard.accuracy.value
    mets_accuracy_thr = None
    mets_accuracy_thresh = None
    mets_precision = metrics_dashboard.precision.value
    mets_recall = metrics_dashboard.recall.value
    mets_dice = None

    acc_code = str('accuracy')
    err_code = str('error_rate')
    thr_code = str('accuracy_thresh')
    k_code = str('top_k_accuracy')
    pre_code = str('precision')
    rec_code = str('recall')
    dice_code = str('dice')

    mets_list=[]
    mets_list_code = []
    output_pres = Precision()
    output_recall = Recall()

    if mets_error == 'Yes':
        mets_list.append(error_rate)
        mets_list_code.append(err_code)
    else:
        None
    if mets_accuracy == 'Yes':
        mets_list.append(accuracy)
        mets_list_code.append(acc_code)
    else:
        None
    if mets_accuracy_thresh == 'Yes':
        mets_list.append(accuracy_thresh)
        mets_list_code.append(thr_code)
    else:
        None
    if mets_accuracy_thr == 'Yes':
        k = data.c
        mets_list.append(top_k_accuracy)
        mets_list_code.append(k_code)
    else:
        None
    if mets_precision == 'Yes':
        mets_list.append(output_pres)
        mets_list_code.append(pre_code)
    else:
        None
    if mets_recall == 'Yes':
        mets_list.append(output_recall)
        mets_list_code.append(rec_code)
    else:
        None
    if mets_dice == 'Yes':
        mets_list.append(dice)
        mets_list_code.append(dice_code)
    else:
        None

    return mets_list, mets_list_code

