import torch
import os

import numpy as np
import random


import json
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
    

def wordify(string):
    word = string.replace('_', ' ')
    return word
 
def replace_underscores(classnames):
    return [classname.replace('_', ' ') for classname in classnames]
 
def replace_commas(descriptions):
    return [description.replace(',', ' ') for description in descriptions]
 
def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
    
def modify_descriptor(descriptor, apply_changes=True):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def load_gpt_descriptions(json_file, classnames, dataname):
    gpt_descriptions = load_json(json_file)
    unmodify_dict = {}
    
    # Filter descriptions based on classnames
    filtered_descriptions = {k: v for k, v in gpt_descriptions.items() if k in replace_underscores(classnames)}
    
    for i, (k, v) in enumerate(filtered_descriptions.items()):
        if len(v) == 0:
            v = ['']
            
        word_to_add = wordify(k)
        if dataname == 'Caltech101':
            prefix = "a photo of a"
            suffix = ""
        if dataname == 'OxfordFlowers':
            prefix = "a photo of a"
            suffix = "a type of flower, "
        if dataname == 'FGVCAircraft':
            prefix = "a photo of a"
            suffix = "a type of aircraft, "
        if dataname == 'UCF101':
            prefix = "a photo of a person doing"
            suffix = ""
        if dataname == 'OxfordPets':
            prefix = "a photo of a"
            suffix = "a type of pet, "
        if dataname == 'Food101':
            prefix = "a photo of a"
            suffix = "a type of food, "
        if dataname == 'DescribableTextures':
            prefix = "a photo of a"
            suffix = "a type of texture, "
        if dataname =='StanfordCars':
            prefix = "a photo of a"
            suffix = ""
        if dataname == 'SUN397':
            prefix = "a photo of a"
            suffix = ""  
            
        if dataname == 'PACS_artpainting':
            prefix = "an art painting of a"
            suffix = "" 
        if dataname == 'PACS_cartoon':
            prefix = "a cartoon of a"
            suffix = "" 
        if dataname == 'PACS_photo':
            prefix = "a photo of a"
            suffix = "" 
        if dataname == 'PACS_sketch':
            prefix = "a sketch of a"
            suffix = ""  
            
        if dataname == 'OfficeHome_art':
            prefix = "an art of a"
            suffix = "" 
        if dataname == 'OfficeHome_clipart':
            prefix = "a clipart of a"
            suffix = "" 
        if dataname == 'OfficeHome_product':
            prefix = "a product image of a"
            suffix = "" 
        if dataname == 'OfficeHome_realworld':
            prefix = "a realworld image of a"
            suffix = ""
            
        if dataname == 'VLCS_CALTECH':
            prefix = "a high quality photo of a"
            suffix = "as a standalone object, " 
        if dataname == 'VLCS_LABELME':
            prefix = "a realworld photo of the"
            suffix = "" 
        if dataname == 'VLCS_PASCAL':
            prefix = "a realworld photo of a"
            suffix = "" 
        if dataname == 'VLCS_SUN':
            prefix = "a photo of a"
            suffix = "in diverse scenic environments, "           
        if dataname == 'DomainNet_clipart':
            prefix = "a clipart of a"
            suffix = "" 
        if dataname == 'DomainNet_infograph':
            prefix = "an infograph of a"
            suffix = "" 
        if dataname == 'DomainNet_painting':
            prefix = "a painting of a"
            suffix = ""  
        if dataname == 'DomainNet_quickdraw':
            prefix = "a quickdraw image of a"
            suffix = ""  
        if dataname == 'DomainNet_real':
            prefix = "a real image of a"
            suffix = ""
        if dataname == 'DomainNet_sketch':
            prefix = "a sketch of a"
            suffix = ""
            
        if dataname == 'TerraIncognita_L38':
            prefix = "a photo of a"
            suffix = "" 
            
        if dataname == 'TerraIncognita_L43':
            prefix = "a photo of a"
            suffix = ""
            
        if dataname == 'TerraIncognita_L46':
            prefix = "a photo of a"
            suffix = ""
            
        if dataname == 'TerraIncognita_L100':
            prefix = "a photo of a"
            suffix = "" 
            
        build_descriptor_string = lambda item: f"{prefix} {word_to_add}, {suffix}{modify_descriptor(item)}"
        unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
        
        filtered_descriptions[k] = [build_descriptor_string(item) for item in v]
    
    return filtered_descriptions


def load_descriptions(json_file, classnames, dataname):
    gpt_descriptions = load_json(json_file)
    unmodify_dict = {}
    
    filtered_descriptions = {k: v for k, v in gpt_descriptions.items() if k in replace_underscores(classnames)}
    
    for i, (k, v) in enumerate(filtered_descriptions.items()):
        if len(v) == 0:
            v = ['']
            
        word_to_add = wordify(k)  
        build_descriptor_string = lambda item: f"{item}"
        unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
        filtered_descriptions[k] = [build_descriptor_string(item) for item in v]

    return filtered_descriptions


def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
 
import matplotlib.pyplot as plt

stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
  
def show_single_image(image):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    denorm_image = denormalize(image.unsqueeze(0).cpu(), *stats)
    ax.imshow(denorm_image.squeeze().permute(1, 2, 0).clamp(0,1))
    
    plt.show()