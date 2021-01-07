## TAG GEN Training Visual GUI

- This is version 1 of our TAG GEN Training GUI.
- Goal is for the training team to be able to utilise this gui to train each attribute to its highest possible accuracy in the easiest visual way.


## How to start:

- To train an attribute, you must begin with the attribute data which will is expected to be in the form of 'attribute' -> 'train'
- Dashboard 1 of the training gui does the following
  - Set the data path as the attribute folder (containing a train folder only), and then use the buttons to:
  - 1) convert to jpg, 2) split into train and valid, 3) clean the data of bad images, 4) augment the data 
- Dashboard 2 
  - No changes required here unless you would like to work with smaller batch size depending on your machine. Stick with the defaults if you can.
- Dashboard 3
  - Set the metrics you'd like to train with. 
  - Accuracy is the default. You can also add precision and recall if required
- Dashboard 4
  - Finally train the model.
  - Begin by using the LR finder. 
  - View the graph generated and figure out which LR would be suitable. 
  - Click train, select the LR you just chose based on the graph, set the number of epochs to train for and then train.
  - Repeat this process until you have reached highest possible accuracy.
  - Specific training steps / examples / instructions will be provided in the notebook.
- Dashboard 5
  - Use this dashboard to check Test Accuracy of the model you've just created (or any model you'd like), on any specific test set.
  - This test accuracy and model predictions will be saved to a csv in the directory of your attribute data.