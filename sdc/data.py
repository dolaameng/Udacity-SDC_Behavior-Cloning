"""
This implements a DataSet class that provides a customized batch generator to model.
"""

import copy
from os import path
from functools import partial
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


class DataSet(object):
    headers = ["CenterImage", "LeftImage", "RightImage", 
            "SteeringAngle", "Throttle", "Break", "Speed"]

    def __init__(self, log_img_paths):
        """
        log_img_paths: a list of tuple (path_to_log.csv, path_to_IMG_folder)
        """
        logs = []
        for log_path, image_folder in log_img_paths:
            #image_folder = path.abspath(image_folder)
            log = pd.read_csv(log_path, header=None, names=self.headers)
            for col in ["CenterImage", "LeftImage", "RightImage"]:
                log[col] = log[col].str.rsplit("/", n=1).str[-1].apply(lambda p: path.join(image_folder, p))
            logs.append(log)
        
        self.log = pd.concat(logs, axis=0, ignore_index=True)
        self.index = 0

    def preprocess(self):
        """preprocess dataset"""
        self.remove_noise()
        # using mirror translation makes training for more epochs possible
        self.mirror()
        self.shuffle()
        return self

    def mirror(self):
        """Mirror the center image and minus the steering"""
        mirror = self.log.copy()
        mirror["CenterImage"] = mirror["CenterImage"] + "_mirror"
        mirror["SteeringAngle"] = - mirror["SteeringAngle"].astype(np.float32)
        self.log = pd.concat([self.log, mirror], axis=0, ignore_index=True)
        return self

    def smooth(self, window):
        """
        smooth steering by rolling mean, so that similiar images (next to each in time)
        won't have steer values that are too different
        """
        self.log.SteeringAngle = pd.rolling_mean(self.log.SteeringAngle, window=window, center=True)
        self.log.SteeringAngle = self.log.SteeringAngle.fillna(0)
        print("steering angle has been smoothed based on window %d" % window)
        return self

    def remove_noise(self):
        N = self.log.shape[0]
        # focus on speed >= 20 
        self.log = self.log[self.log.Speed >= 20.]
        print("%d records have been removed due to speed <= 20" % (N- self.log.shape[0]))
        return self

    def shuffle(self):
        self.log = self.log.reindex(np.random.permutation(self.log.index))
        return self

    def copy_constructor(self, log, index):
        rhs = copy.deepcopy(self)
        rhs.log = log
        rhs.index = index
        return rhs

    def split(self, test_size):
        itrain, itest = train_test_split(np.arange(self.log.shape[0]), test_size=test_size)
        train_log = self.log.iloc[itrain]
        test_log = self.log.iloc[itest]
        train_dataset = self.copy_constructor(train_log, 0)
        test_dataset = self.copy_constructor(test_log, 0)
        return (train_dataset, test_dataset)

    def reset(self):
        self.index = 0
        return self

    def make_batch_generator(self, batch_size, col_grps, process_fns = None):
        """
        col_grps = list of grouped cols, each grouped cols are a list/string itself, e.g.
        col_grps = [xcols, ycols, wtcols]. Data from multiple cols in a group are stacked

        process_fns = a dict of {col: fn}, where fn is used to process col. The fn must take a single 
        element (not a batch), e.g, an image as input. 
        """
        process_fns = process_fns or {}
        def _generator(stream):
            batch_items = []
            for i, row in enumerate(stream):
                item = []
                for cols in col_grps:
                    col_list = cols if type(cols) is list else [cols]
                    output = row[col_list]
                    for icol, col in enumerate(col_list):
                        if col in process_fns:
                            fn = process_fns[col]
                            output[icol] = fn(output[icol])
                    if type(cols) == list:
                        output = np.stack(output, axis=0)
                    else:
                        output = output[0]
                    item.append(output)
                batch_items.append(item)
                if len(batch_items) >= batch_size:
                    current_batch = batch_items[:batch_size]
                    #yield(zip(*current_batch))
                    yield tuple(map(np.asarray, zip(*current_batch)))
                    batch_items = batch_items[batch_size:]
        return _generator(self)

    def size(self):
        return self.log.shape[0]

    def __next__(self):
        self.index %= self.size()
        self.index += 1
        return self.log.iloc[self.index-1]

    def __iter__(self):
        return self

    def next(self):
        return __next__(self)