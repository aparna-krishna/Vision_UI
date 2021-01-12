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


import webbrowser
from IPython.display import YouTubeVideo
from ipywidgets import HBox, Label

import warnings
warnings.filterwarnings('ignore')


import fastai
import psutil



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


###########################################

## Modules for loading files from folders ##

###########################################


def button_f():
    """Helper for folder_choices"""
    print('>> Do you need to specify train and/or valid folder locations:')
    print('>> Leave unchecked for default fastai values')

    button_fs = widgets.Button(description='Confirm')

    button_f.train = widgets.Checkbox(
        value=False,
        description='Specify Train folder location',
        disabled=False
        )
    button_f.valid = widgets.Checkbox(
        value=False,
        description='Specify Valid folder location',
        disabled=False
        )
    ui = widgets.HBox([button_f.train, button_f.valid])
    display(ui)

    display(button_fs)

    out = widgets.Output()
    display(out)

    def on_button_clicked(b):
        with out:
            clear_output()
            folder_choices()
    button_fs.on_click(on_button_clicked)

def folder_choices():
    """Helper for in_folder choices"""
    button_fc = widgets.Button(description='Choice')
    button_tv = widgets.Button(description='Train and Valid folder')
    button_v = widgets.Button(description='Valid Folder')
    button_t = widgets.Button(description='Train folder')

    if button_f.train.value == True and button_f.valid.value == True:
        ui = widgets.HBox([button_tv])

    elif button_f.train.value == False and button_f.valid.value == True:
        ui = widgets.HBox([button_v])

    elif button_f.train.value == True and button_f.valid.value == False:
        ui = widgets.HBox([button_t])

    else:
        ui = None
        print("Using default values of 'train' and 'valid' folders")
        in_folder_train.path = 'train'
        in_folder_valid.path = 'valid'
        pct_metrics()

    out = widgets.Output()
    display(out)

    display(ui)

    def on_button_clicked_tv(b):
        clear_output()
        in_folder_train()
        in_folder_valid()
        print(f'Train folder: {in_folder_train.path}')
        print(f'Valid folder: {in_folder_valid.path}')
        pct_metrics()
    button_tv.on_click(on_button_clicked_tv)

    def on_button_clicked_t(b):
        clear_output()
        in_folder_train()
        print(f'Train folder: {in_folder_train.path}')
        in_folder_valid.path = 'valid'
        pct_metrics()
    button_t.on_click(on_button_clicked_t)

    def on_button_clicked_v(b):
        clear_output()
        in_folder_valid()
        print(f'Valid folder: {in_folder_valid.path}\n')
        in_folder_train.path = 'train'
        pct_metrics()
    button_v.on_click(on_button_clicked_v)



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
        
        path = ds.data_path.value

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

        arch_work()

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
                    a, b = metrics_list(mets_list, mets_list_code)
                    opt_func = partial(RAdam, betas=(0.9,0.99), eps=1e-6)
                    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, opt_func=opt_func, custom_head=None)

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
                        a, b = metrics_list(mets_list, mets_list_code)
                        b_ = b[0]
                        learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=a, custom_head=None)
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


  

def arch_work():
    arch_work.info = models.resnet50

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

            print(f'Data in: {ds.data_path.value}| Normalization: {architecture_dashboard.norma.value}| Architecture: {architecture_dashboard.archi.value}| Pretrain: {architecture_dashboard.pretrain_check.value}|'
            f'Batch Size: {architecture_dashboard.f.value}| Image Size: {architecture_dashboard.m.value}')

            print(f'Training Metrics: {metrics_list(mets_list)} ')

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
    print ('Chosen metrics: ',metrics_list(mets_list))
    metrics_list(mets_list)

    
    batch_val = int(architecture_dashboard.f.value) # batch size
    image_val = int(architecture_dashboard.m.value) # image size

    r = architecture_dashboard.pretrain_check.value
    t = metrics_list(mets_list)

    path = ds.data_path

    tfms= get_transforms(do_flip=True, max_lighting=0.01, max_rotate=10,
                      max_zoom=1.05,p_lighting=0.,p_affine=0.1, max_warp=0.01)
    
    # assumes split valid etc all created, needs a test file
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=batch_val, size=image_val, test='test')

    learn = cnn_learner(data, base_arch=arch_work.info, pretrained=r, metrics=metrics_list(mets_list,mets_list_code), custom_head=None)

    lr_out = widgets.Output()
    display(lr_out)

    with lr_out:
        learn.lr_find()     
        learn.recorder.plot()



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
    save_result = ds.data_path.value +'/gui_PredResults.csv'
    accuracy=get_test_accuracy.get_accuracy(results_dashboard.testset, results_dashboard.model, save_result, 
    results_dashboard.true_values, results_dashboard.attribute)
    out_res = widgets.Output()
    display(out_res)
    with out_res:
        print("Accuracy for 'silhouette' attribute on given test-set is:",accuracy,"%")
        print()
        print('Model predictions saved at:',save_result)


#############################

### MAIN DISPLAY FUNCTION ###

#############################

def display_ui():
    """ Display tabs for visual display"""
    button = widgets.Button(description="Train")
    button_b = widgets.Button(description="Metrics")
    button_m = widgets.Button(description='Model')
    button_l = widgets.Button(description='LR')

    test_button = widgets.Button(description='Batch')
    test2_button = widgets.Button(description='Test2')

    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()

    data1 = pd.DataFrame(np.random.normal(size = 165))
    data2 = pd.DataFrame(np.random.normal(size = 85))
    data5 = pd.DataFrame(np.random.normal(size = 245))
    data6 = pd.DataFrame(np.random.normal(size = 325))
    data7 = pd.DataFrame(np.random.normal(size= 405))


    with out1:          #preprocess
        clear_output()
        ds()
    
    with out2:          #arhictecture
        clear_output()
        architecture_dashboard()

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
                arch_work()
                metrics_dashboard()
                #get_data()

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
                get_data()
        button_tr.on_click(on_button_clicked)

    with out5:      # results
         results_dashboard()
   

    display_ui.tab = widgets.Tab(children = [out1, out2, out3, out4, out5])
    display_ui.tab.set_title(0, 'PreProcess')
    display_ui.tab.set_title(1, 'Architecture')
    display_ui.tab.set_title(2, 'Metrics')
    display_ui.tab.set_title(3, 'Train')
    display_ui.tab.set_title(4, 'Results')
    display(display_ui.tab)
