"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`. 
Currently supported datasets:
- ImageNet (:class:`robustness.datasets.ImageNet`)
- RestrictedImageNet (:class:`robustness.datasets.RestrictedImageNet`)
- CIFAR-10 (:class:`robustness.datasets.CIFAR`)
- CINIC-10 (:class:`robustness.datasets.CINIC`)
- A2B: horse2zebra, summer2winter_yosemite, apple2orange
  (:class:`robustness.datasets.A2B`)
:doc:`../example_usage/training_lib_part_2` shows how to add custom
datasets to the library.
"""

import pathlib

import torch as ch
import torch.utils.data
from torchvision import transforms, datasets

from dataset.tools import constants
from dataset.tools import openimgs_helpers
from dataset.tools import data_augmentation as da
from dataset.tools import loaders

from dataset.tools.helpers import get_label_mapping


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std', 
                         'transform_train', 'transform_test']
        optional_args = ['custom_class', 'label_mapping', 'custom_class_args']

        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args} 

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)
    
    def override_args(self, default_args, kwargs):
        '''
        Convenience method for overriding arguments. (Internal)
        '''
        for k in kwargs:
            if not (k in default_args): continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, 
                    subset_start=0, subset_type='rand', val_batch_size=None,
                    only_val=False, shuffle_train=True, shuffle_val=True, subset_seed=None,
                    poison_ratio=0.1, 
                    target=0, 
                    patch_size=64,
                    black_trigger=False,
                    distribution=False):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.
        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:
            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128) 
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val,
                                    seed=subset_seed,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val,
                                    custom_class_args=self.custom_class_args,
                                    poison_ratio=poison_ratio, 
                                    target=target, 
                                    patch_size=patch_size,
                                    black_trigger=black_trigger,
                                    distributed=distribution)

class RestrictedImageNet(DataSet):
    '''
    RestrictedImagenet Dataset [TSE+19]_
    A subset of ImageNet with the following labels:
    * Dog (classes 151-268)
    * Cat (classes 281-285)
    * Frog (classes 30-32)
    * Turtle (classes 33-37)
    * Bird (classes 80-100)
    * Monkey (classes 365-382)
    * Fish (classes 389-397)
    * Crab (classes 118-121)
    * Insect (classes 300-319)
    To initialize, just provide the path to the full ImageNet dataset
    (no special formatting required).
    .. [TSE+19] Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., &
        Madry, A. (2019). Robustness May Be at Odds with Accuracy. ICLR
        2019.
    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'restricted_imagenet'
        ds_kwargs = {
            'num_classes': len(constants.RESTRICTED_IMAGNET_RANGES),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                constants.RESTRICTED_IMAGNET_RANGES),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(RestrictedImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

