"""
Model testing utils

Created on: Tuesday, April 27th, 2021
Author: Jacob A Rose


"""

from torch import nn
import collections




from pl_bolts.utils import BatchGradientVerification

def perform_batch_gradient_verification(model, input_size=(3, 224, 224), input_array=None):
    """
    Checks if a model mixes data across the batch dimension.
    
    This can happen if reshape- and/or permutation operations are carried out in the wrong order or
    on the wrong tensor dimensions.
    
    Examples:
    ========
    perform_batch_gradient_verification(model)

    perform_batch_gradient_verification(model, input_size=(3, 224, 224))

    perform_batch_gradient_verification(model, input_array=torch.rand(2,3,224,224))
    
    """
    if input_array is None:
        input_array = model.example_input_array
    if input_array is None:
        input_array = torch.rand(2,*input_size)
    verification = BatchGradientVerification(model)
    valid = verification.check(input_array=input_array, sample_idx=1)
    
    if valid:
        print('Test: [PASSED]',
              '\n==============\n',
              'Model passed batch gradient verification test!\n',
              'Confirmed no data mixing occurs across batch dimension')
    
    return valid


if __name__=="__main__":
    
    backbone_name = 'resnet18'
    model = timm.create_model(backbone_name, pretrained=True)
    
    valid = perform_batch_gradient_verification(model)










# # source: https://github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/utils/torch_utils.py
# def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict):
#     """
#     Copy weights from state dict to model, skipping layers that are incompatible.
#     This method is helpful if you are doing some model surgery and want to load
#     part of the model weights into different model.
#     :param model: Model to load weights into
#     :param model_state_dict: Model state dict to load weights from
#     :return: None
#     """
#     for name, value in model_state_dict.items():
#         try:
#             model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
#         except Exception as e:
#             print(e)
