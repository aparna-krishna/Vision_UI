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
from dashboards import preprocess_dashboard, metrics_dashboard
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


############################

## Test Results Dashboard

############################


def results_dashboard():

    style = {'description_width': 'initial'}
    ## load the model you just trained, specify test folder path & print test accuracy
    #print('>> 1) Specify path to your model (file selection opens in new window)' '\n')
    
    results_dashboard.model_path = widgets.Textarea(placeholder='path to trained attribute model .pkl file',disabled=False, layout=widgets.Layout(width='50%', height='30px'))
    model_btn = widgets.Button(description='Confirm', button_style='info', style=style)
    enter_path = HBox([Label( "Path to attribute model:", layout=widgets.Layout(width='20%', height='30px')), results_dashboard.model_path, model_btn])
    display(enter_path)
    print()


    #print('>> 2)  Select a test dataset')
    results_dashboard.testset_path = widgets.Textarea(placeholder='path to your folder containing test images',disabled=False, layout=widgets.Layout(width='50%', height='30px'))
    test_btn = widgets.Button(description='Confirm', button_style='info', style=style)
    enter_path2 = HBox([Label('Path to test dataset:',layout=widgets.Layout(width='20%', height='30px')),results_dashboard.testset_path, test_btn])
    display(enter_path2)
    print()

    #print('>> 3) Enter path to true values for your test datatset')
    results_dashboard.trueval_path = widgets.Textarea(placeholder='path to csv file containing true values of test set',disabled=False, layout=widgets.Layout(width='50%', height='30px'))
    trueval_btn = widgets.Button(description='Confirm', button_style='info', style=style)
    enter_path3 = HBox([Label("Path to test's true_values:",layout=widgets.Layout(width='20%', height='30px')),results_dashboard.trueval_path, trueval_btn])
    display(enter_path3)
    print()

    #print('>> 4) Enter the attribute name you wish to find accuracy for: (must correspond to column name is true_values csv)')
    results_dashboard.attr_name = widgets.Textarea(placeholder='enter attribute name (must correspond to a column name in your true values csv)',disabled=False, layout=widgets.Layout(width='50%', height='30px'))
    attr_btn = widgets.Button(description='Confirm', button_style='info', style=style)
    enter_path4 = HBox([Label("Enter attribute name:",layout=widgets.Layout(width='20%', height='30px')),results_dashboard.attr_name, attr_btn])
    display(enter_path4)
    print()



    testdata_accuracy_btn = widgets.Button(description='Calculate', button_style='info')
    enter_path5 = HBox([Label("Get attribute's test accuracy:",layout=widgets.Layout(width='20%', height='30px')), testdata_accuracy_btn])
    display(enter_path5)


    out = widgets.Output()
    display(out)

    def on_model_button(b):
        with out:
            #clear_output()
            results_dashboard.model = results_dashboard.model_path.value
    
    def on_testdata_button(b):
        with out:
            #clear_output()
            results_dashboard.testset = results_dashboard.testset_path.value
    
    def on_accuracy_button(b):
        with out:
            #lear_output()
            calculate_test_accuracy()

    def on_trueval_button(b):
        with out:
            #lear_output()
            results_dashboard.true_values = results_dashboard.trueval_path.value

    def on_attr_button(b):
        with out:
            #lear_output()
            results_dashboard.attribute = results_dashboard.attr_name.value
    
    model_btn.on_click(on_model_button)
    test_btn.on_click(on_testdata_button)
    testdata_accuracy_btn.on_click(on_accuracy_button)
    trueval_btn.on_click(on_trueval_button)
    attr_btn.on_click(on_attr_button)

def calculate_test_accuracy():
    # TODO, load learning, use ur predict_fn function and get accuracy
    # where to save results
    # true values
    # attribute name
    save_result = preprocess_dashboard.ds.data_path.value +'/gui_PredResults.csv'
    accuracy=get_test_accuracy.get_accuracy(results_dashboard.testset, results_dashboard.model, save_result, 
    results_dashboard.true_values, results_dashboard.attribute)
    out_res = widgets.Output()
    display(out_res)
    with out_res:
        print("Accuracy for 'silhouette' attribute on given test-set is:",accuracy,"%")
        print()
        print('Model predictions saved at:',save_result)

