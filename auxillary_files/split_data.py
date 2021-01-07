import os,shutil,random


def create_validation_set(train_dir, validation_dir):

        labels = os.listdir(train_dir)
        percentage = 20

        for label in labels:
                if not label.startswith('.'): 
                        train_label_dir = os.path.join(train_dir, str(label))
                        num_files = len([name for name in os.listdir(train_label_dir)])

                        validation_label_dir = os.path.join(validation_dir, str(label))
                        if not os.path.exists(validation_label_dir):
                                os.makedirs(validation_label_dir)
                        validation_num_files = int(percentage*num_files/100)

                        #print(train_label_dir, validation_label_dir, num_files, validation_num_files)
                        
                        
                        for i in range(validation_num_files):
                                file_to_move = random.choice(os.listdir(train_label_dir))
             
                                shutil.move(os.path.join(train_label_dir,file_to_move),os.path.join(validation_label_dir,file_to_move))
       

                

#train_dir ='/Users/aparna/Desktop/data_tagGen/test_for_gui2/train' #train_dir = "train"
#validation_dir = '/Users/aparna/Desktop/data_tagGen/test_for_gui2/valid'



#create_validation_set(train_dir, os.listdir(train_dir), validation_dir,20)


