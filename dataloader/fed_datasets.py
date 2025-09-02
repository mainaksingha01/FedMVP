from dataloader.utils import *
import os
import pickle
from collections import OrderedDict

class Caltech101(DatasetBase):
    dataset_dir = "caltech-101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "Caltech101"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)


class OxfordFlowers(DatasetBase):
    dataset_dir = "oxford_flowers"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OxfordFlowers"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        
        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class EuroSAT(DatasetBase):
    dataset_dir = "eurosat"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "EuroSAT"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class OxfordPets(DatasetBase):
    dataset_dir = "oxford_pets"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OxfordPets"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)


class FGVCAircraft(DatasetBase):
    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "FGVCAircraft"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train, train_cnames = self.read_data(cname2lab, "images_variant_train.txt")
        val, val_cnames = self.read_data(cname2lab, "images_variant_val.txt")
        test, test_cnames = self.read_data(cname2lab, "images_variant_test.txt")
    
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items, cnames = [], []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                cnames.append(classname)
        return items, cnames


class Food101(DatasetBase):
    dataset_dir = "food-101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "Food101"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class DescribableTextures(DatasetBase):
    dataset_dir = "dtd"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DescribableTextures"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class UCF101(DatasetBase):
    dataset_dir = "ucf101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "UCF101"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class StanfordCars(DatasetBase):
    dataset_dir = "stanford_cars"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "StanfordCars"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.dataset_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class SUN397(DatasetBase):
    dataset_dir = "sun397"

    def __init__(self, cfg,available_classes=None,relabel=True):
        self.data_name = "SUN397"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")
        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
        
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


class ImageNet(DatasetBase):
    dataset_dir = "imagenet"

    def __init__(self, cfg,available_classes=None,relabel=True):
        self.data_name = 'ImageNet'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
                train_cnames = preprocessed["train_cnames"]
                test_cnames = preprocessed["test_cnames"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train, train_cnames = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test, test_cnames = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test, "train_cnames": train_cnames, "test_cnames": test_cnames}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        if available_classes is not None:
            output, cnames = subsample_classes(train, test, available_classes=available_classes,relabel=relabel)
            train, test = output[0], output[1]
            train_cnames, test_cnames = cnames[0], cnames[1]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, test_cnames, test_cnames

        print('Imagenet is loaded.')

        super().__init__(train=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames
TO_BE_IGNORED = ["README.txt","class_to_idx.json",""]

class ImageNetA(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-adversarial"

    def __init__(self, cfg):
        self.data_name = 'ImageNetA'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir =os.path.join(self.dataset_dir, "imagenet-a")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetR(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"
    def __init__(self, cfg):
        self.data_name = 'ImageNetR'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt", "class_to_idx.json", "dataset.h5"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetSketch(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"
    def __init__(self, cfg):
        self.data_name = 'ImageNetSketch'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt", "class_to_idx.json", "dataset.h5"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetV2(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """
    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        self.data_name = 'ImageNetV2'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []
        all_cnames = []
        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items,all_cnames
    
class PACS_artpainting(DatasetBase):
    dataset_dir = "pacs"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "PACS_artpainting"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "art_painting")
        self.split_path = os.path.join(self.dataset_dir, "art_painting", "split_pacs_artpainting.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)    
    
class PACS_cartoon(DatasetBase):
    dataset_dir = "pacs"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "PACS_cartoon"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "cartoon")
        self.split_path = os.path.join(self.dataset_dir, "cartoon", "split_pacs_cartoon.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        

class PACS_photo(DatasetBase):
    dataset_dir = "pacs"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "PACS_photo"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "photo")
        self.split_path = os.path.join(self.dataset_dir, "photo", "split_pacs_photo.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class PACS_sketch(DatasetBase):
    dataset_dir = "pacs"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "PACS_sketch"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "sketch")
        self.split_path = os.path.join(self.dataset_dir, "sketch", "split_pacs_sketch.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class OfficeHome_art(DatasetBase):
    dataset_dir = "officehome"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OfficeHome_art"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "art")
        self.split_path = os.path.join(self.dataset_dir, "art", "split_officehome_art.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class OfficeHome_clipart(DatasetBase):
    dataset_dir = "officehome"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OfficeHome_clipart"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "clipart")
        self.split_path = os.path.join(self.dataset_dir, "clipart", "split_officehome_clipart.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class OfficeHome_product(DatasetBase):
    dataset_dir = "officehome"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OfficeHome_product"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "product")
        self.split_path = os.path.join(self.dataset_dir, "product", "split_officehome_product.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class OfficeHome_realworld(DatasetBase):
    dataset_dir = "officehome"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OfficeHome_realworld"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "realworld")
        self.split_path = os.path.join(self.dataset_dir, "realworld", "split_officehome_realworld.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class VLCS_CALTECH(DatasetBase):
    dataset_dir = "VLCS"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "VLCS_CALTECH"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "CALTECH/full")
        self.split_path = os.path.join(self.dataset_dir, "CALTECH", "split_VLCS_CALTECH.json")
        
        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class VLCS_LABELME(DatasetBase):
    dataset_dir = "VLCS"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "VLCS_LABELME"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "LABELME/full")
        self.split_path = os.path.join(self.dataset_dir, "LABELME", "split_VLCS_LABELME.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class VLCS_PASCAL(DatasetBase):
    dataset_dir = "VLCS"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "VLCS_PASCAL"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "PASCAL/full")
        self.split_path = os.path.join(self.dataset_dir, "PASCAL", "split_VLCS_PASCAL.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class VLCS_SUN(DatasetBase):
    dataset_dir = "VLCS"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "VLCS_SUN"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN/full")
        self.split_path = os.path.join(self.dataset_dir, "SUN", "split_VLCS_SUN.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_clipart(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_clipart"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "clipart")
        self.split_path = os.path.join(self.dataset_dir, "clipart", "split_domainnet_clipart.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_infograph(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_infograph"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "infograph")
        self.split_path = os.path.join(self.dataset_dir, "infograph", "split_domainnet_infograph.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_painting(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_painting"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "painting")
        self.split_path = os.path.join(self.dataset_dir, "painting", "split_domainnet_painting.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_quickdraw(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_quickdraw"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "quickdraw")
        self.split_path = os.path.join(self.dataset_dir, "quickdraw", "split_domainnet_quickdraw.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_real(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_real"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "real")
        self.split_path = os.path.join(self.dataset_dir, "real", "split_domainnet_real.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
class DomainNet_sketch(DatasetBase):
    dataset_dir = "domainnet"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DomainNet_sketch"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "sketch")
        self.split_path = os.path.join(self.dataset_dir, "sketch", "split_domainnet_sketch.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
        
class TerraIncognita_L38(DatasetBase):
    dataset_dir = "terra_incognita"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "TerraIncognita_L38"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "location_38")
        self.split_path = os.path.join(self.dataset_dir, "location_38", "split_terraincognita_location38.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
        
class TerraIncognita_L43(DatasetBase):
    dataset_dir = "terra_incognita"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "TerraIncognita_L43"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "location_43")
        self.split_path = os.path.join(self.dataset_dir, "location_43", "split_terraincognita_location43.json")
        # print(self.split_path)
        # exit()
        # /home/mainak/data/terra_incognita/location_43/split_terraincognita_location43.json
        

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
        
class TerraIncognita_L46(DatasetBase):
    dataset_dir = "terra_incognita"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "TerraIncognita_L46"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "location_46")
        self.split_path = os.path.join(self.dataset_dir, "location_46", "split_terraincognita_location46.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)
        
        
class TerraIncognita_L100(DatasetBase):
    dataset_dir = "terra_incognita"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "TerraIncognita_L100"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "location_100")
        self.split_path = os.path.join(self.dataset_dir, "location_100", "split_terraincognita_location100.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)