''' ipywidgets '''
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets
import ipywidgets as widgets
from IPython.display import display,clear_output
from ipywidgets import HBox, Label

''' fast ai, data related '''
import pandas as pd

from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import*
from fastai.widgets import ClassConfusion
import fastai
import psutil

import matplotlib.pyplot as plt

''' import the dashboards and preprocessing scripts '''
import auxillary_files.webp_to_jpg_v1 as webp_to_jpg_v1
import auxillary_files.split_data as split_data
import auxillary_files.get_test_accuracy as get_test_accuracy
import auxillary_files.check_image_sanity as check_image_sanity
import auxillary_files.Data_Aug_v2 as Data_Aug_v2
from auxillary_files.radam import RAdam
from dashboards import preprocess_dashboard, architecture_dashboard, metrics_dashboard, testResult_dashboard, train_dashboard


import warnings
warnings.filterwarnings('ignore')

#############################

### Display the dashboards ###

#############################

def display_ui():

    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()


    with out1:          #preprocess
        clear_output()
        data_path = preprocess_dashboard.ds()
    

    with out2:          #arhictecture
        clear_output()
        architecture_dashboard.architecture_dashboard()

    with out3:          #Metrics
        button_m = widgets.Button(description="Metrics", button_style='info')
        metrics = HBox([Label("Click to select metrics:",layout=widgets.Layout(width='20%', height='30px')),button_m])
        display(metrics)

        print()
        print()
        out = widgets.Output()
        display(out)

        def on_button_clicked_learn(b):
            with out:
                clear_output()
                architecture_dashboard.arch_work()
                metrics_dashboard.metrics_dashboard()

        button_m.on_click(on_button_clicked_learn)

    with out4:       # train
        button_tr = widgets.Button(description='Train')
        display(button_tr)
        print ('>> Click to view training parameters and learning rate''\n''\n')
        out_tr = widgets.Output()
        display(out_tr)
        def on_button_clicked(b):
            with out_tr:
                clear_output()
                train_dashboard.get_data()
        button_tr.on_click(on_button_clicked)

    with out5:      # results
         testResult_dashboard.results_dashboard()
   

    display_ui.tab = widgets.Tab(children = [out1, out2, out3, out4, out5])
    display_ui.tab.set_title(0, 'PreProcess')
    display_ui.tab.set_title(1, 'Architecture')
    display_ui.tab.set_title(2, 'Metrics')
    display_ui.tab.set_title(3, 'Train')
    display_ui.tab.set_title(4, 'Results')
    display(display_ui.tab)
