from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
from pathlib import Path

from ..dataset import ImageDataset


class Fy24(ImageDataset):

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))

        train, query, gallery = self.process_dir(self.root)

        super(Fy24, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        root_path = Path(dir_path)

        PATTERN = re.compile(R"camera([^/]+)/rgb/[^/]+/([0-9]+)/.*")
        # camera22-direct/rgb/2024-11-01/2242689/2024-11-01_15-07-21_2024-11-01_15-23-12_67000_655/2024-11-01_15-07-19_291.png

        train = []
        query = []
        gallery = []
        total_cnt = 0
        pid_map = {}
        train_pid_map = {}

        for pic_ab in root_path.rglob("*.png"):
            pic_re = pic_ab.relative_to(root_path)
            m = PATTERN.match(str(pic_re))
            if not m:
                print(f"discard: {pic_re}")
            camera_name = m[1]
            pid = m[2]

            camid = total_cnt % 6
            pid_remap = pid_map.setdefault(pid, len(pid_map))

            if pid_remap % 2 == 0:
                train_pid = train_pid_map.setdefault(pid_remap, len(train_pid_map))
                triplet = (str(pic_ab), train_pid, camid)
                train.append(triplet)
            else:
                test_cnt = total_cnt - len(train)
                triplet = (str(pic_ab), pid_remap, camid)
                if test_cnt % 3 != 2:
                    # 0, 1
                    gallery.append(triplet)
                else:
                    query.append(triplet)

            total_cnt += 1

        return train, query, gallery