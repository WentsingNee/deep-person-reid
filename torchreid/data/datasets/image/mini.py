from __future__ import division, print_function, absolute_import

import random
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class Mini(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))

        # allow alternative directory structure
        self.data_dir = self.root

        train, query, gallery = self.process_dir(self.data_dir)
        super(Mini, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        label2pid = {}

        train = []
        query = []
        gallery = []
        for by_label_path in glob.glob(osp.join(dir_path, "*")):
            if not osp.isdir(by_label_path):
                continue
            label = osp.relpath(by_label_path, dir_path)
            label2pid[label] = (pid := len(label2pid) + 1)

            for idx, img_path in enumerate(glob.glob(osp.join(by_label_path, "*.png"))):
                camera_id = random.randint(1, 6)
                if pid % 2 == 0:
                    train.append((img_path, pid, camera_id))
                elif idx % 3 == 0:
                    query.append((img_path, pid, camera_id))
                else:
                    gallery.append((img_path, pid, camera_id))

        return train, query, gallery
