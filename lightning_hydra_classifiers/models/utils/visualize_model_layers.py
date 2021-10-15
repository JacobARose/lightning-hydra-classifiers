"""

lightning_hydra_classifiers/models/utils/visualize_model_layers.py

Description: Defines a helper class to conveniently log the outputs of various model layers using register_forward_hook()

Created on: Tuesday, October 12th, 2021
Author: Jacob A Rose


"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms as T
from typing import *
from torchinfo import summary

from tqdm.auto import trange, tqdm
import collections
# from torch import nn



def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to('cpu').numpy()    


class SaveOutput:
    """
    Simple container class to be used by LayerOutputsImageRecorder
    Maintains a list of model outputs produced during inference after an instance of this class has been registered as a forward hook.
    """
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
        
        
class LayerOutputsImageRecorder:
    
    """
    Available methods:
    
    __init__
    adjust_plot_kwargs
    clear
    get_device
    
    register_model
    register_image
    display_layers
    display_layer
    
    TBD:
        - Create a clean looking repr for the class
        - Allow configurable saving workflows optimized for specific analysis needs (e.g. format outpaths into subdirectories for each input image vs. each layer)
        - Improve the channel visualization -- adapt # of axes to # of channels.
        
    configs (TBD):
        - 
            i. compare models & images side-by-side:
                - root/layer_{0}/{img_names[0]}__{model_names.A}.jpg
                - root/layer_{0}/{img_names[0]}__{model_names.B}.jpg
                - root/layer_{0}/{img_names[1]}__{model_names.A}.jpg
                - root/layer_{0}/{img_names[1]}__{model_names.B}.jpg
                - root/...
                - root/layer_{0}/{img_names[m]}__{model_names.A}.jpg
                - root/layer_{0}/{img_names[m]}__{model_names.B}.jpg
                - root/...
                - root/layer_{n}/{img_names[m]}__{model_names.A}.jpg
                - root/layer_{n}/{img_names[m]}__{model_names.B}.jpg

                
            i. compare models & layers side-by-side:
                - root/{img_names[0]}/layer_{0}__{model_names.A}.jpg
                - root/{img_names[0]}/layer_{0}__{model_names.B}.jpg
                - root/...
                - root/{img_names[0]}/layer_{n}__{model_names.A}.jpg
                - root/{img_names[0]}/layer_{n}__{model_names.B}.jpg
                - root/...
                - root/{img_names[m]}/layer_{0}__{model_names.A}.jpg
                - root/{img_names[m]}/layer_{0}__{model_names.B}.jpg
                - root/...
                - root/{img_names[m]}/layer_{n}__{model_names.A}.jpg
                - root/{img_names[m]}/layer_{n}__{model_names.B}.jpg
                



    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 save_dir: Optional[Union[Path, str]]=None,
                 save_conv_outputs: bool=True,
                 save_model_summary: bool=True,
                 model_name: str="",
                 img_path: Optional[Union[str, List[str]]]=None,
                 img_tensor: Optional[torch.Tensor]=None,
                 img_size: Tuple[int]=(224,224)):
        self._saver = SaveOutput()
        self.clear()
        self.adjust_plot_kwargs()
        self.save_dir = save_dir
        self.img_size = img_size
        self.register_model(model, model_name, save_conv_outputs, save_model_summary)
        self.register_image(img_path, img_tensor, img_size)

    def clear(self):
        self._saver.clear()
        self.clear_hooks()
        self.hook_handles = []
        self.layer_names = []
        self.output_paths = []
        for attr in ["img", "img_path", "img_tensor", "img_size", "output_tensor"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def get_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        
    def register_model(self, 
                       model: torch.nn.Module,
                       model_name: str="",
                       save_conv_outputs: bool=True,
                       save_model_summary: bool=True):
        self.model = model
        self.device = self.get_device()
        self.model_name = model_name
        self.save_conv_outputs = save_conv_outputs
        self.save_model_summary = save_model_summary

        if save_model_summary:
            self.model_summary = log_model_summary(model=model,
                                                   input_size=(1,3,*self.img_size),
                                                   full_summary=True,
                                                   working_dir=self.save_dir,
                                                   model_name=model_name)
        
        modules = list(self.model.named_modules())
        for i, (name, layer) in enumerate(modules):
            if self.save_conv_outputs and isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self._saver)
                self.hook_handles.append(handle)
                self.layer_names.append(name)
#                 print(f"Layer {name} ({i}/{len(modules)}): Registered forward_hook")

    def clear_hooks(self):
        if hasattr(self, "hook_handles") and isinstance(getattr(self, "hook_handles"), list):
            for h in range(len(self.hook_handles)):
                self.hook_handles[0].remove()
                del self.hook_handles[0]
            
        
#     def register_image(self,
#                        path: Optional[str]=None,
#                        img_tensor: Optional[torch.Tensor]=None,
#                        img_size: Tuple[int]=(224,224)):
#         self.img_size = img_size
#         self.input_transform = T.Compose([T.Resize(self.img_size), T.ToTensor()])
        
#         if os.path.isfile(str(path)):
#             self.img_path = path
#             self.img = Image.open(path)
#             self.img.save(os.path.join(self.save_dir, f"original_{Path(path).stem}.jpg"))
#             self.img_tensor = self.input_transform(self.img).unsqueeze(dim=0).to(self.model.device)
#         elif img_tensor is not None:
#             self.img_path = None
#             self.img = None
#             self.img_tensor = img_tensor
#             T.ToPILImage()(self.img).save(os.path.join(self.save_dir, f"original_input_image.jpg"))
#         else:
#             raise ArgumentError("Must pass either a path to an on-disk image file or a ready-to-use image tensor.")
            
#         self.output_tensor = self.model(self.img_tensor)


#     def adjust_plot_kwargs(self, reset_to_defaults: Optional[bool]=False, **kwargs):
#         if reset_to_defaults or (not hasattr(self, "plot_kwargs")):
#             self.plot_kwargs = {"figsize": (20, 20),
#                                 "gridsize": (4, 4),
#                                 "rect": (0.0, 0.0, 1.0, 0.97)}
# #                                 "rect": (0.0, 0.0, 1.0, 0.97)}
#         self.plot_kwargs.update(kwargs)


#     def display_layer(self,
#                       layer_index: int=0,
#                       img_index: int=0,
#                       save_dir: Optional[Union[Path, str]]=None,
#                       figtitle: Union[bool, str]=False,
#                       verbose: bool=True):
#         """
#         Display the output arrays from layers[layer_index] arranged in a 4x4 grid & optionally save them to disk.
#         """
        
#         model_name = self.model_name
#         img_name = "default"
#         if isinstance(self.img_path, (Path, str)):
#             img_name = Path(self.img_path).stem

#         output_imgs = tensor2np(self._saver.outputs[layer_index])

#         if verbose: print(f"Displaying first 16 channels from the output of layer #{layer_index}")
            
#         #TODO move the ugly code here into a separate plot image grid function
#         with plt.style.context("seaborn-white"):
#             plt.figure(figsize=self.plot_kwargs["figsize"], frameon=False, constrained_layout=True)
#             max_channels_to_plot = np.prod(self.plot_kwargs['gridsize'])
#             for idx in trange(max_channels_to_plot, desc=f"Channels:", position=1):
#                 plt.subplot(*self.plot_kwargs["gridsize"], idx+1)
#                 plt.imshow(output_imgs[img_index, idx])
#             plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

# #             plt.tight_layout(self.plot_kwargs["rect"])
#             if figtitle is True:
#                 figtitle = f"Image: {img_name}|Model: {model_name}|Layer: {self.layer_names[layer_index]}, shape(B,C,H,W)={output_imgs.shape}|first {max_channels_to_plot} channels"
#             if isinstance(figtitle, str): plt.suptitle(figtitle, fontsize="x-large") #medium")
            
#             if (save_dir is None) and (self.save_dir is not None):
#                 save_dir = self.save_dir    
#             if isinstance(save_dir, (Path, str)):
# #                 outpath = os.path.join(save_dir, model_name, f"layer_{layer_index}", img_name + ".jpg")
#                 outpath = os.path.join(save_dir, f"layer_{layer_index}", img_name + ".jpg")
#                 os.makedirs(Path(outpath).parent, exist_ok=True)
#                 plt.savefig(outpath)
#                 if verbose:
#                     print(f"Exported conv layer outputs to image file: {outpath}")
#                 self.output_paths.append(outpath)


    def register_image(self,
                       path: Optional[str]=None,
                       img_tensor: Optional[torch.Tensor]=None,
                       img_size: Tuple[int]=(224,224)):
        self.img_size = img_size
        self.input_transform = T.Compose([T.Resize(self.img_size), T.ToTensor()])
        if isinstance(path, list):
            if os.path.isfile(path[0]):
                self.img_path = path
                self.img = [Image.open(p) for p in path]
                self.img_tensor = torch.stack([self.input_transform(img) for img in self.img]).to(self.model.device)
        elif os.path.isfile(str(path)):
            self.img_path = path
            self.img = Image.open(path)
            self.img_tensor = self.input_transform(self.img).to(self.model.device)
        elif img_tensor is not None:
            self.img_path = None
            self.img = None
            self.img_tensor = img_tensor            
        else:
            raise ArgumentError("Must pass either a path to an on-disk image file or a ready-to-use image tensor.")
            
        if self.img_tensor.ndimension() == 3:
            self.img_tensor = self.img_tensor.unsqueeze(dim=0)
        if self.img_tensor.ndimension() == 4:
            self.num_imgs = self.img_tensor.shape[0]
        self.output_tensor = self.model(self.img_tensor)


    def adjust_plot_kwargs(self, reset_to_defaults: Optional[bool]=False, **kwargs):
        if reset_to_defaults or (not hasattr(self, "plot_kwargs")):
            self.plot_kwargs = {"figsize": (20, 20),
                                "gridsize": (4, 4),
                                "rect": (0.0, 0.0, 1.0, 0.97)}
        self.plot_kwargs.update(kwargs)


    def display_layer(self,
                      layer_index: int=0,
                      img_index: int=0,
                      save_dir: Optional[Union[Path, str]]=None,
                      figtitle: Union[bool, str]=False,
                      verbose: bool=True):
        """
        Display the output arrays from layers[layer_index] arranged in a 4x4 grid & optionally save them to disk.
        """
        
        model_name = self.model_name
        img_name = "default"
        if isinstance(self.img_path, (Path, str)):
            img_name = Path(self.img_path).stem

        output_imgs = tensor2np(self._saver.outputs[layer_index])

        if verbose: print(f"Displaying first 16 channels from the output of layer #{layer_index}")
            
        #TODO move the ugly code here into a separate plot image grid function
        with plt.style.context("seaborn-white"):
#             for img_idx in trange(self.num_imgs, desc="input imgs:", position=1):
            plt.figure(figsize=self.plot_kwargs["figsize"], frameon=False, constrained_layout=True)
            max_channels_to_plot = np.prod(self.plot_kwargs['gridsize'])
            for idx in trange(max_channels_to_plot, desc=f"Channels:", position=2):
                plt.subplot(*self.plot_kwargs["gridsize"], idx+1)
#                     plt.imshow(output_imgs[0, idx])
                plt.imshow(output_imgs[img_index, idx])
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

#             plt.tight_layout(self.plot_kwargs["rect"])
            if figtitle is True:
                figtitle = f"Image: {img_name}|Model: {model_name}|Layer: {self.layer_names[layer_index]}, shape(B,C,H,W)={output_imgs.shape}|first {max_channels_to_plot} channels"

            if isinstance(figtitle, str): plt.suptitle(figtitle, fontsize="x-large") #medium")

            if (save_dir is None) and (self.save_dir is not None):
                save_dir = self.save_dir    
            if isinstance(save_dir, (Path, str)):
#                 outpath = os.path.join(save_dir, model_name, f"layer_{layer_index}", img_name + ".jpg")
                outpath = os.path.join(save_dir, f"layer_{layer_index}", img_name + ".jpg")
                os.makedirs(Path(outpath).parent, exist_ok=True)
                plt.savefig(outpath)
                if verbose:
                    print(f"Exported conv layer outputs to image file: {outpath}")
                self.output_paths.append(outpath)
            plt.close()

                
    def display_layers(self,
                      layer_range: Optional[Union[int, Sequence]]=None,
                      save_dir: Optional[Union[Path, str]]=None,
                      figtitle: Union[bool, str]=False,
                      verbose: bool=True):
        """
        Call recorder.display_layer on the set of model layers indicated in layer_range.
        """
        layer_range = self.get_indices(index=layer_range)
        
        for idx in tqdm(layer_range, desc=f"Layers:", total=len(layer_range), position=0):
            for img_idx in trange(self.num_imgs, desc="input imgs:", position=1):
                self.display_layer(layer_index=idx,
                                   img_index=img_idx,
                                   save_dir=save_dir,
                                   figtitle=figtitle,
                                   verbose=verbose)
            
        return self.output_paths
    
    def _get_index(self, index: int=None) -> int:
        max_layers = len(self.hook_handles)
        if index is None:
            index = max_layers
        if isinstance(index, int):
            if index < 0:
                index = max_layers + index
        return index
        
    def get_indices(self, index: Union[int, Sequence]) -> List:
        if (index is None) or isinstance(index, int):
            return list(range(self._get_index(index)))
        if isinstance(index, Sequence):
            index = list(index)
            for i in range(len(index)):
                index[i] = self._get_index(index[i])
        else:
            raise ArgumentError("index must be either an int, Sequence-like type, or None.")
        return index

    
    

def log_model_summary(model: torch.nn.Module,
                      input_size: Tuple[int],
                      full_summary: bool=True,
                      working_dir: str=".",
                      model_name: Optional[str]=None,
                      verbose: bool=1):
    """
    produce a text file with the model summary
    
    TODO: Add this to Eval Plugins
    
    log_model_summary(model=model,
                  working_dir=working_dir,
                  input_size=(1, data_config.channels, *data_config.image_size),
                  full_summary=True)

    """

    if full_summary:
        col_names=("kernel_size", "input_size","output_size", "num_params", "mult_adds")
    else:
        col_names=("input_size","output_size", "num_params")

    model_summary = summary(model.cuda(),
                            input_size=input_size,
                            row_settings=('depth', 'var_names'),
                            col_names=col_names,
                            verbose=verbose)

    if (model_name in ("", None)) and (hasattr(model, "name")):
        model_name = model.name
    if (model_name in ("", None)):
        summary_path = os.path.join(working_dir, f'model_summary.txt')
    else:
        summary_path = os.path.join(working_dir, f'{model_name}_model_summary.txt')
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(str(model_summary))
        
    return model_summary


#############################################################
#############################################################
            


def build_parser():
    DEFAULT_IMG_PATH = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/3/jpg/Hydrangeaceae/Hydrangeaceae_Fendlerella_utahensis_Hickey_Hickey_6028.jpg"
    DEFAULT_MODEL = "resnet34"
    DEFAULT_DEV_DIR = os.environ.get('DEV_DIR', "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/experiment_logs/dev")
    DEFAULT_IMG_RESOLUTION = 224
    
    
    parser = argparse.ArgumentParser(description='visualize model layer outputs using forward hooks')
    parser.add_argument('-img', '--image-path', dest="img_path", default=DEFAULT_IMG_PATH)
    parser.add_argument('-res', '--image-resolution', dest="resolution", default=DEFAULT_IMG_RESOLUTION)
    parser.add_argument('--model', dest="model", default=DEFAULT_MODEL,
                        help="Load an [optionally] imagenet-pretrained model from torchvision.models to demonstrate visualization of layer outputs.")
    parser.add_argument('--save-dir', dest="save_dir", default=DEFAULT_DEV_DIR)
    parser.add_argument("-cpu", "--cpu-only", dest="cpu_only", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", default=True)
    
    return parser
        
            
def main(argv=None):
    
    parser = build_parser()
    args = parser.parse_args(argv)
    args.img_size = (args.resolution, args.resolution)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    from torchvision import models
    MODEL_FACTORY = getattr(models, args.model)
    model = MODEL_FACTORY(pretrained=args.pretrained)
    model = model.to(device)
    model.device = device

#     img_path = "/media/data/jacob/GitHub/lightning-hydra-classifiers/tests/test_images/wide_aspect_ratio_leaf_image.jpg"
#     model_name = "resnet34"
    if args.model not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.model)
        
    hook_recorder = LayerOutputsImageRecorder(model=model,
                                           save_dir=args.save_dir,
                                           save_conv_outputs=True,
                                           save_model_summary=True,
                                           model_name=args.model,
                                           img_path=args.img_path,
                                           img_size=args.img_size)


    max_hooks = len(hook_recorder.hook_handles)
    num_hooks = 12

    hook_recorder.display_layers(layer_range=range(0, max_hooks, max_hooks//num_hooks),
                                 save_dir=args.save_dir,
                                 figtitle=True)
    
    
import pytorch_lightning as pl
    
class LayerOutputsImageRecorderCallback(pl.Callback):
    
    def __init__(self,
                 save_dir: Optional[Union[Path, str]]=None,
                 save_conv_outputs: bool=True,
                 save_model_summary: bool=True,
                 model_name: str="",
                 img_size: Tuple[int]=(224,224),
                 num_imgs: int=5):

        super().__init__()
        self.save_dir = save_dir
        self.save_conv_outputs = save_conv_outputs
        self.save_model_summary = save_model_summary
        self.model_name = model_name
#         self.img_path = img_path
#         self.img_tensor = img_tensor
        self.img_size = img_size
        self.num_imgs = num_imgs
        self._has_setup_train = False
        self._has_setup_val = False
        self._has_setup_test = False
    
    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage=stage)
        if self._has_setup_train and stage in ("train", "fit"):
            return
        if self._has_setup_val and stage == "val":
            return
        
        if not self._has_setup_test:
            self._has_setup_test = False
            return
        
        self.dataset = trainer.datamodule.get_dataset(stage=stage)
        img_paths = [self.dataset.fetch_item(i)[0] for i in range(self.num_imgs)]
        self.recorder = LayerOutputsImageRecorder(model=pl_module,
                                                  save_dir=self.save_dir,
                                                  save_conv_outputs=seelf.save_conv_outputs,
                                                  save_model_summary=self.save_model_summary,
                                                  model_name=self.model_name,
                                                  img_path=img_paths,
                                                  img_tensor=None,
                                                  img_size=self.img_size):
    
        self._has_setup_train = True
        self._has_setup_val = True
        self._has_setup_test = True

        
    for idx in range(args.):
        hook_recorder.display_layer(idx, figtitle=True)
        
        
        
if __name__ == "__main__":
        
    main()
