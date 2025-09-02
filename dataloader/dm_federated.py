import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torch.utils.data import Sampler
from dataloader.fed_datasets import *
import numpy as np
from clip.clip import _transform
import random
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
import random
import math



class ClassSampler(Sampler):
    def __init__(self, data_cnames, classes, k, seed):

        self.classes = classes
        self.data_cnames = data_cnames
        self.k = k
        self.seed = seed

        class_to_indices = self._create_class_to_indices()
        indices = []
        for indices_per_class in class_to_indices.values():
            indices += indices_per_class
        self.indices = indices

    def _create_class_to_indices(self):
        if self.seed >= 0:
            # np.random.seed(self.seed)
            random.seed(self.seed)
        class_to_indices = {c: [] for c in self.classes}
        for c in tqdm(class_to_indices.keys()):
            valid_idx = np.where(np.array(self.data_cnames)==c)[0].tolist()
            try:
                sample_idx = random.sample(valid_idx,self.k)
            except:
                sample_idx = valid_idx
            class_to_indices[c] = sample_idx

        return class_to_indices

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_data_loader(
        cfg,
        data_source,
        data_cnames,
        classes=None,
        batch_size=64,
        num_shots=None,
        tfm=None,
):
    dataset_wrapper = DatasetWrapper(data_source, transform=tfm)
    # Build sampler
    sampler = None
    if classes is not None:
        sampler = ClassSampler(
            data_cnames,
            classes,
            num_shots,
            cfg.SEED
        )

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader



class TrainDataManager:

    def __init__(
            self,
            cfg,
            dataname,
            available_classes
    ):
        # Load dataset
        if dataname == 'caltech101':
            dataset = Caltech101(cfg,available_classes,relabel=True)

        elif dataname == 'oxford_flowers':
            dataset = OxfordFlowers(cfg,available_classes,relabel=True)
        elif dataname == 'eurosat':
            dataset = EuroSAT(cfg,available_classes,relabel=True)
        elif dataname == 'oxford_pets':
            dataset = OxfordPets(cfg,available_classes,relabel=True)

        elif dataname == 'fgvc_aircraft':
            dataset = FGVCAircraft(cfg,available_classes,relabel=True)

        elif dataname == 'food101':
            dataset = Food101(cfg,available_classes,relabel=True)

        elif dataname == 'dtd':
            dataset = DescribableTextures(cfg,available_classes,relabel=True)

        elif dataname == 'ucf101':
            dataset = UCF101(cfg,available_classes,relabel=True)

        elif dataname == 'stanford_cars':
            dataset = StanfordCars(cfg,available_classes,relabel=True)

        elif dataname == 'sun397':
            dataset = SUN397(cfg,available_classes,relabel=True)

        elif dataname == 'imagenet':
            dataset = ImageNet(cfg,available_classes,relabel=True)
            
        elif dataname == 'pacs_artpainting':
            dataset = PACS_artpainting(cfg,available_classes,relabel=True)
        
        elif dataname == 'pacs_cartoon':
            dataset = PACS_cartoon(cfg,available_classes,relabel=True)
        
        elif dataname == 'pacs_photo':
            dataset = PACS_photo(cfg,available_classes,relabel=True)
        
        elif dataname == 'pacs_sketch':
            dataset = PACS_sketch(cfg,available_classes,relabel=True)
            
        elif dataname == 'officehome_art':
            dataset = OfficeHome_art(cfg,available_classes,relabel=True)
                
        elif dataname == 'officehome_clipart':
            dataset = OfficeHome_clipart(cfg,available_classes,relabel=True)
                
        elif dataname == 'officehome_product':
            dataset = OfficeHome_product(cfg,available_classes,relabel=True)
                
        elif dataname == 'officehome_realworld':
            dataset = OfficeHome_realworld(cfg,available_classes,relabel=True)
            
        elif dataname == 'vlcs_caltech':
            dataset = VLCS_CALTECH(cfg,available_classes,relabel=True)
                
        elif dataname == 'vlcs_labelme':
            dataset = VLCS_LABELME(cfg,available_classes,relabel=True)
                
        elif dataname == 'vlcs_pascal':
            dataset = VLCS_PASCAL(cfg,available_classes,relabel=True)
                
        elif dataname == 'vlcs_sun':
            dataset = VLCS_SUN(cfg,available_classes,relabel=True)
            
        elif dataname == 'domainnet_clipart':
            dataset = DomainNet_clipart(cfg,available_classes,relabel=True)
                
        elif dataname == 'domainnet_infograph':
            dataset = DomainNet_infograph(cfg,available_classes,relabel=True)
                
        elif dataname == 'domainnet_painting':
            dataset = DomainNet_painting(cfg,available_classes,relabel=True)
                
        elif dataname == 'domainnet_quickdraw':
            dataset = DomainNet_quickdraw(cfg,available_classes,relabel=True)
            
        elif dataname == 'domainnet_real':
            dataset = DomainNet_real(cfg,available_classes,relabel=True)
                
        elif dataname == 'domainnet_sketch':
            dataset = DomainNet_sketch(cfg,available_classes,relabel=True)
            
        elif dataname == 'terra_incognita_l38':
            dataset = TerraIncognita_L38(cfg,available_classes,relabel=True)
            
        elif dataname == 'terra_incognita_l43':
            dataset = TerraIncognita_L43(cfg,available_classes,relabel=True)
            
        elif dataname == 'terra_incognita_l46':
            dataset = TerraIncognita_L46(cfg,available_classes,relabel=True)
            
        elif dataname == 'terra_incognita_l100':
            dataset = TerraIncognita_L100(cfg,available_classes,relabel=True)

        tfm = _transform(224)


        # Build train_loader

        train_loader = build_data_loader(
            cfg,
            data_source=dataset.train,
            data_cnames = dataset.train_cnames,
            classes=dataset.classnames,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            num_shots=cfg.DATASET.NUM_SHOTS,
            tfm=tfm,
        )

        test_loader = build_data_loader(
            cfg,
            data_source=dataset.test,
            data_cnames = dataset.test_cnames,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm,
        )
        self.train_loader = train_loader
        self.available_classes = dataset.classnames
        self.data_name = dataset.data_name
        self.test_loader = test_loader




class TestDataManager:

    def __init__(
            self,
            cfg,
            split
    ):
        dataset_classnum = {'imagenet': 1000, 'caltech101':100, 'oxford_flowers': 102,'eurosat':10, 'oxford_pets':37, 'fgvc_aircraft': 100,
                            'food101': 101, 'dtd': 47, 'ucf101':101,'stanford_cars':196,'sun397':397 ,'imagenet-a': 200,'imagenet-s': 200,'imagenet-r': 200,'imagenet-v2': 200,
                            'pacs_artpainting': 7, 'pacs_cartoon': 7, 'pacs_photo': 7, 'pacs_sketch': 7,
                            'officehome_art': 65, 'officehome_clipart': 65, 'officehome_product': 65, 'officehome_realworld': 65,
                            'vlcs_caltech': 5, 'vlcs_labelme': 5, 'vlcs_pascal': 5, 'vlcs_sun': 5,
                            'domainnet_clipart': 345, 'domainnet_infograph': 345, 'domainnet_painting': 345, 'domainnet_quickdraw': 345, 'domainnet_real': 345, 'domainnet_sketch': 345,
                            'terra_incognita_l38':10, 'terra_incognita_l43':10, 'terra_incognita_l46':10, 'terra_incognita_l100':10}
        # Load dataset
        available_datasets = cfg.DATASET.TESTNAME_SPACE
        # split = cfg.TEST.SPLIT
        test_loaders, test_datasets = [],[]
        for dataname in available_datasets:
            all_cls_idx = np.arange(dataset_classnum[dataname])
            m = math.ceil(dataset_classnum[dataname] / 2)
            if split == 'base':
                available_classes = all_cls_idx[:m]
            elif split == 'new':
                available_classes = all_cls_idx[m:]
            else:
                available_classes = None
            if dataname == 'caltech101':

                dataset = Caltech101(cfg,available_classes)

            elif dataname == 'oxford_flowers':
                dataset = OxfordFlowers(cfg,available_classes)
            elif dataname == 'eurosat':
                dataset = EuroSAT(cfg,available_classes)
            elif dataname == 'oxford_pets':
                dataset = OxfordPets(cfg,available_classes)

            elif dataname == 'fgvc_aircraft':
                dataset = FGVCAircraft(cfg,available_classes)

            elif dataname == 'food101':
                dataset = Food101(cfg,available_classes)

            elif dataname == 'dtd':
                dataset = DescribableTextures(cfg,available_classes)

            elif dataname == 'ucf101':
                dataset = UCF101(cfg,available_classes)

            elif dataname == 'stanford_cars':
                dataset = StanfordCars(cfg,available_classes)

            elif dataname == 'sun397':
                dataset = SUN397(cfg,available_classes)

            elif dataname == 'imagenet':
                dataset = ImageNet(cfg,available_classes)
            elif dataname == 'imagenet-a':
                dataset = ImageNetA(cfg)
            elif dataname == 'imagenet-r':
                dataset = ImageNetR(cfg)
            elif dataname == 'imagenet-s':
                dataset = ImageNetSketch(cfg)
            elif dataname == 'imagenet-v2':
                dataset = ImageNetV2(cfg)
                
            elif dataname == 'pacs_artpainting':
                dataset = PACS_artpainting(cfg,available_classes,relabel=True)
        
            elif dataname == 'pacs_cartoon':
                dataset = PACS_cartoon(cfg,available_classes,relabel=True)
        
            elif dataname == 'pacs_photo':
                dataset = PACS_photo(cfg,available_classes,relabel=True)
        
            elif dataname == 'pacs_sketch':
                dataset = PACS_sketch(cfg,available_classes,relabel=True)
                
            elif dataname == 'officehome_art':
                dataset = OfficeHome_art(cfg,available_classes,relabel=True)
                
            elif dataname == 'officehome_clipart':
                dataset = OfficeHome_clipart(cfg,available_classes,relabel=True)
                
            elif dataname == 'officehome_product':
                dataset = OfficeHome_product(cfg,available_classes,relabel=True)
                
            elif dataname == 'officehome_realworld':
                dataset = OfficeHome_realworld(cfg,available_classes,relabel=True)
                
            elif dataname == 'vlcs_caltech':
                dataset = VLCS_CALTECH(cfg,available_classes,relabel=True)
                
            elif dataname == 'vlcs_labelme':
                dataset = VLCS_LABELME(cfg,available_classes,relabel=True)
                
            elif dataname == 'vlcs_pascal':
                dataset = VLCS_PASCAL(cfg,available_classes,relabel=True)
                
            elif dataname == 'vlcs_sun':
                dataset = VLCS_SUN(cfg,available_classes,relabel=True)
                
            elif dataname == 'domainnet_clipart':
                dataset = DomainNet_clipart(cfg,available_classes,relabel=True)
                
            elif dataname == 'domainnet_infograph':
                dataset = DomainNet_infograph(cfg,available_classes,relabel=True)
                
            elif dataname == 'domainnet_painting':
                dataset = DomainNet_painting(cfg,available_classes,relabel=True)
                
            elif dataname == 'domainnet_quickdraw':
                dataset = DomainNet_quickdraw(cfg,available_classes,relabel=True)
            
            elif dataname == 'domainnet_real':
                dataset = DomainNet_real(cfg,available_classes,relabel=True)
                
            elif dataname == 'domainnet_sketch':
                dataset = DomainNet_sketch(cfg,available_classes,relabel=True)
                
            elif dataname == 'terra_incognita_l38':
                dataset = TerraIncognita_L38(cfg,available_classes,relabel=True)
            
            elif dataname == 'terra_incognita_l43':
                dataset = TerraIncognita_L43(cfg,available_classes,relabel=True)
                
            elif dataname == 'terra_incognita_l46':
                dataset = TerraIncognita_L46(cfg,available_classes,relabel=True)
                
            elif dataname == 'terra_incognita_l100':
                dataset = TerraIncognita_L100(cfg,available_classes,relabel=True)
                
            tfm = _transform(224)


            test_loader = build_data_loader(
                cfg,
                data_source=dataset.test,
                data_cnames=dataset.test_cnames,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm,
            )
            test_loaders.append(test_loader)
            test_datasets.append(dataset)
        self.test_loaders = test_loaders
        self.test_datasets = test_datasets


class DatasetWrapper(TorchDataset):

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        label = item.label
        classname = item.classname
        try:
            img = Image.open(item.impath).convert("RGB")
        except:
            img = item.impath

        if self.transform is not None:
            img = self.transform(img)

        output = {
            "img": img,
            "label": label,
            "cname":classname
        }
        return output


