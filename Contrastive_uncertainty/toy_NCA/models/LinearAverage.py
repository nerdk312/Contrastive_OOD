import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    
    @staticmethod
    def forward(self, x, y, memory, params):
        ''' In the forward pass, we receive a tensor containing
        the input and return a tensor containing the output
        '''
        T = params[0].item()  # Nawid - Temperature term of softmax
        batchSize = x.size(0)  # Nawid - batch size

        # inner product
        out = torch.mm(x.data, memory.t())  # Nawid similarity between the data and the information in the memory bank
        out.div_(T) # batchSize * N  # Nawid - scale by temperature
        # Nawid - cache objects for use in backward pass
        self.save_for_backward(x, memory, y, params)  # Nawid -  save for backward pass

        return out

    ''' In the backward pass we receive a tensor containing the graident 
    of the loss with respect to the output, and we need to compute
    the gradient of the loss with respect to the input

    '''
    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors  # Nawid - tensors which were saved
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()  # Nawid - encoder momentum to update the memory bank I believe
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # Nawid - updating the memory bank
        weight_pos.mul_(momentum)  
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        # Nawid - params are temperatire and memory bank momentum,
        self.register_buffer('params',torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize/3)
        # Nawid - initialising the memory bank
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out