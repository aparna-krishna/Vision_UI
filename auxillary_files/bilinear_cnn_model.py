from fastai import *
from fastai.vision import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class bilinearCNN_Model(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the model with number of classes which will be important to define the output layer
        """
        super(bilinearCNN_Model, self).__init__()
        
        self.epsilon = 1e-8
        self.body = create_body(arch = models.resnet50)
        self.fc = nn.Linear(2048 * 2048, num_classes)

    def forward(self, X):
        """
        Define how forward pass should occur here
        """
        
        conv_op = self.body(X)
        
        # Cross product operation
        # Refer section 3.1.1 in this paper https://arxiv.org/pdf/1504.07889.pdf
        conv_op = conv_op.view(conv_op.size(0), 2048, 8 * 8)
        conv_op_Transposed = torch.transpose(conv_op, 1, 2)
        bilinearMatrix = torch.bmm(conv_op, conv_op_Transposed) / (8 * 8)
        flattened = bilinearMatrix.view(bilinearMatrix.size(0), 2048 * 2048)
        
        # Refer section 3.1.2 in this paper https://arxiv.org/pdf/1504.07889.pdf
        # The signed square root
        flattened = torch.sign(flattened) * torch.sqrt(torch.abs(flattened) + self.epsilon)
        
        # Refer section 3.1.3 in this paper https://arxiv.org/pdf/1504.07889.pdf
        # L2 regularization
        op = F.normalize(flattened)

        out = self.fc(op)

        return out
    
    def predictBatch(self, X):
        """
        Define how the predictions should happen in a batch
        """
        with torch.no_grad():
            predictions = self.forward(X)
           
        return predictions