import numpy as np
import glob
from pathlib import Path
from os.path import join, exists
import logging
import open3d as o3d

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)

# original_label_to_names = {
#     0: 'unclassified',
#     100000000: 'other',
#     200000000: 'surface',
#     201000000: 'other-surface',
#     202000000: 'ground',
#     202010000: 'other-ground',
#     202020000: 'road',
#     202030000: 'sidewalk',
#     202040000: 'curb',
#     202050000: 'island',
#     202060000: 'vegetation',
#     203000000: 'building',
#     300000000: 'object',
#     301000000: 'other-object',
#     302000000: 'static',
#     302010000: 'other-static',
#     302020000: 'punctual-object',
#     302020100: 'other-punctual',
#     302020200: 'post',
#     302020300: 'bollard',
#     302020400: 'floor-lamp',
#     302020500: 'traffic-light',
#     302020600: 'traffic-sign',
#     302020700: 'signboard',
#     302020800: 'mailbox',
#     302020900: 'trash-can',
#     302021000: 'meter',
#     302021100: 'bicycle-terminal',
#     302021200: 'bicycle-rack',
#     302021300: 'statue',
#     302030000: 'linear',
#     302030100: 'other-linear',
#     302030200: 'barrier',
#     302030300: 'roasting',
#     302030400: 'grid',
#     302030500: 'chain',
#     302030600: 'wire',
#     302030700: 'low-wall',
#     302040000: 'extended',
#     302040100: 'other-extended',
#     302040200: 'shelter',
#     302040300: 'kiosk',
#     302040400: 'scaffolding',
#     302040500: 'bench',
#     302040600: 'distribution-box',
#     302040700: 'lighting-console',
#     302040800: 'windmill',
#     303000000: 'dynamic',
#     303010000: 'other-dynamic',
#     303020000: 'pedestrian',
#     303020100: 'other-pedestrian',
#     303020200: 'still-pedestrian',
#     303020300: 'walking-pedestrian',
#     303020400: 'running-pedestrian',
#     303020500: 'stroller-pedestrian',
#     303020600: 'holding-pedestrian',
#     303020700: 'leaning-pedestrian',
#     303020800: 'skater',
#     303020900: 'rollerskater',
#     303021000: 'wheelchair',
#     303030000: '2-wheelers',
#     303030100: 'other-2-wheels',
#     303030200: 'bicycle',
#     303030201: 'other-bicycle',
#     303030202: 'mobile-bicycle',
#     303030203: 'stopped-bicycle',
#     303030204: 'parked-bicycle',
#     303030300: 'scooter',
#     303030301: 'other-scooter',
#     303030302: 'mobile-scooter',
#     303030303: 'stopped-scooter',
#     303030304: 'parked-scooter',
#     303030400: 'moped',
#     303030401: 'other-moped',
#     303030402: 'mobile-moped',
#     303030403: 'stopped-moped',
#     303030404: 'parked-moped',
#     303030500: 'motorbike',
#     303030501: 'other-motorbike',
#     303030502: 'mobile-motorbike',
#     303030503: 'stopped-motorbike',
#     303030504: 'parked-motorbike',
#     303040000: '4+-wheelers',
#     303040100: 'other-4+-wheels',
#     303040200: 'car',
#     303040201: 'other-car',
#     303040202: 'mobile-car',
#     303040203: 'stopped-car',
#     303040204: 'parked-car',
#     303040300: 'van',
#     303040301: 'other-van',
#     303040302: 'mobile-van',
#     303040303: 'stopped-van',
#     303040304: 'parked-van',
#     303040400: 'truck',
#     303040401: 'other-truck',
#     303040402: 'mobile-truck',
#     303040403: 'stopped-truck',
#     303040404: 'parked-truck',
#     303040500: 'bus',
#     303040501: 'other-bus',
#     303040502: 'mobile-bus',
#     303040503: 'stopped-bus',
#     303040504: 'parked-bus',
#     303050000: 'furniture',
#     303050100: 'other-furniture',
#     303050200: 'table',
#     303050300: 'chair',
#     303050400: 'stool',
#     303050500: 'trash-can',
#     303050600: 'waste',
#     304000000: 'natural',
#     304010000: 'other-natural',
#     304020000: 'tree',
#     304030000: 'bush',
#     304040000: 'potted-plant',
#     304050000: 'hedge'
# }


# label_to_names = {
#     0: 'unclassified',
#     2: 'surface',
#     3: 'other-surface',
#     4: 'ground',
#     5: 'other-ground',
#     6: 'road',
#     7: 'sidewalk',
#     8: 'curb',
#     9: 'island',
#     10: 'vegetation',
#     11: 'building',
#     12: 'object',
#     13: 'other-object',
#     14: 'static',
#     15: 'other-static',
#     16: 'punctual-object',
#     17: 'other-punctual',
#     18: 'post',
#     19: 'bollard',
#     20: 'floor-lamp',
#     21: 'traffic-light',
#     22: 'traffic-sign',
#     23: 'signboard',
#     24: 'mailbox',
#     25: 'trash-can',
#     26: 'meter',
#     27: 'bicycle-terminal',
#     28: 'bicycle-rack',
#     29: 'statue',
#     30: 'linear',
#     31: 'other-linear',
#     32: 'barrier',
#     33: 'roasting',
#     34: 'grid',
#     35: 'chain',
#     36: 'wire',
#     37: 'low-wall',
#     38: 'extended',
#     39: 'other-extended',
#     40: 'shelter',
#     41: 'kiosk',
#     42: 'scaffolding',
#     43: 'bench',
#     44: 'distribution-box',
#     45: 'lighting-console',
#     46: 'windmill',
#     47: 'dynamic',
#     48: 'other-dynamic',
#     49: 'pedestrian',
#     50: 'other-pedestrian',
#     51: 'still-pedestrian',
#     52: 'walking-pedestrian',
#     53: 'running-pedestrian',
#     54: 'stroller-pedestrian',
#     55: 'holding-pedestrian',
#     56: 'leaning-pedestrian',
#     57: 'skater',
#     58: 'rollerskater',
#     59: 'wheelchair',
#     60: '2-wheelers',
#     61: 'other-2-wheels',
#     62: 'bicycle',
#     63: 'other-bicycle',
#     64: 'mobile-bicycle',
#     65: 'stopped-bicycle',
#     66: 'parked-bicycle',
#     67: 'scooter',
#     68: 'other-scooter',
#     69: 'mobile-scooter',
#     70: 'stopped-scooter',
#     71: 'parked-scooter',
#     72: 'moped',
#     73: 'other-moped',
#     74: 'mobile-moped',
#     75: 'stopped-moped',
#     76: 'parked-moped',
#     77: 'motorbike',
#     78: 'other-motorbike',
#     79: 'mobile-motorbike',
#     80: 'stopped-motorbike',
#     81: 'parked-motorbike',
#     82: '4+-wheelers',
#     83: 'other-4+-wheels',
#     84: 'car',
#     85: 'other-car',
#     86: 'mobile-car',
#     87: 'stopped-car',
#     88: 'parked-car',
#     89: 'van',
#     90: 'other-van',
#     91: 'mobile-van',
#     92: 'stopped-van',
#     93: 'parked-van',
#     94: 'truck',
#     95: 'other-truck',
#     96: 'mobile-truck',
#     97: 'stopped-truck',
#     98: 'parked-truck',
#     99: 'bus',
#     100: 'other-bus',
#     101: 'mobile-bus',
#     102: 'stopped-bus',
#     103: 'parked-bus',
#     104: 'furniture',
#     105: 'other-furniture',
#     106: 'table',
#     107: 'chair',
#     108: 'stool',
#     109: 'trash-can',
#     110: 'waste',
#     111: 'natural',
#     112: 'other-natural',
#     113: 'tree',
#     114: 'bush',
#     115: 'potted-plant',
#     116: 'hedge'
# }

label_to_names = {
    0: 'unlabeled',      # All others
    1: 'sidewalk',            # 202030000
    2: 'road',        # 202020000
    3: 'vegetation',  # 202060000
    4: 'island',         # 202050000
    5: 'building',      # 203000000
    6: 'traffic-sign',   # 302020600
    7: 'traffic-light'           # 302020500
}



COLOR_MAP = {
    0: (0, 0, 0),         # Unlabeled (black)
    1: (255, 127, 80),    # Sidewalk (coral)
    2: (255, 215, 0),     # Road (gold)
    3: (34, 139, 34),     # Vegetation (forest green)
    4: (128, 0, 128),     # Island (purple)
    5: (178, 34, 34),     # Building (firebrick)
    6: (0, 191, 255),     # Traffic-sign (deep sky blue)
    7: (255, 69, 0),       # Traffic-light (red-orange)
    8: (90, 30, 150),
    9: (255, 0, 255),
}



class ParisLille3DExtended(BaseDataset):
    """This class is used to create a dataset based on the ParisLille3D dataset,
    and used in visualizer, training, or testing.

    The ParisLille3D dataset is best used to train models for urban
    infrastructure. You can download the dataset `here <https://npm3d.fr/paris-lille-3d>`__.
    """

    def __init__(self,
                 dataset_path,
                 name='ParisLille3DExtended',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 test_result_folder='./test',
                 val_files=['Lille2.ply'],
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
                dataset_path: The path to the dataset to use.
                name: The name of the dataset (ParisLille3D in this case).
                cache_dir: The directory where the cache is stored.
                use_cache: Indicates if the dataset should be cached.
                num_points: The maximum number of points to use when splitting the dataset.
                ignored_label_inds: A list of labels that should be ignored in the dataset.
                test_result_folder: The folder where the test results should be stored.
                val_files: The files that include the values.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         test_result_folder=test_result_folder,
                         val_files=val_files,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        train_path = cfg.dataset_path + "/training_50_classes/"
        self.train_files = glob.glob(train_path + "/*.ply")
        self.val_files = [
            f for f in self.train_files if Path(f).name in cfg.val_files
        ]
        self.train_files = [
            f for f in self.train_files if f not in self.val_files
        ]

        test_path = cfg.dataset_path + "/test_50_classes/"
        self.test_files = glob.glob(test_path + '*.ply')
        print("Initialized ParisLille3DExtended")

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """

        return label_to_names

    def get_split(self, split):
        return ParisLille3DSplit(self, split=split)
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
	"""

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
                        attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.txt')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.txt')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class ParisLille3DSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = o3d.t.io.read_point_cloud(pc_path).point
        points = pc["positions"].numpy().astype(np.float32)

        if (self.split != 'test'):
            labels = pc["class"].numpy().astype(np.int32).reshape((-1,))
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {'point': points, 'feat': None, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.ply', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(ParisLille3DExtended)
