"""
Waymo dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob
import warnings

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WaymoDataset(DefaultDataset):
    def __init__(
        self,
        timestamp=(0,),
        reference_label=True,
        timing_embedding=False,
        subset_size=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert timestamp[0] == 0
        self.timestamp = timestamp
        self.reference_label = reference_label
        self.timing_embedding = timing_embedding
        self.subset_size = subset_size
        self.data_list = sorted(self.data_list)

        if not (0.0 < self.subset_size <= 1.0):
             raise ValueError(
                 f"subset_size must be between 0.0 (exclusive) and 1.0 (inclusive), but got {self.subset_size}"
             )

        if self.subset_size < 1.0:
            if self.test_mode:
                warnings.warn(
                    f"subset_size ({self.subset_size}) is ignored during test_mode."
                )
            else:
                original_size = len(self.data_list)
                if original_size > 0:
                    num_samples = max(1, int(original_size * self.subset_size))
                    indices = np.linspace(
                        0, original_size - 1, num=num_samples, dtype=int
                    )
                    self.data_list = [self.data_list[i] for i in indices]
                    print(
                        f"Using {len(self.data_list)} samples ({self.subset_size*100:.2f}%) "
                        f"of the original {original_size} samples, selected via linspace."
                    )
                else:
                     warnings.warn("Dataset is empty, subset selection skipped.")

        _, self.sequence_offset, self.sequence_index = np.unique(
            [os.path.dirname(data) for data in self.data_list],
            return_index=True,
            return_inverse=True,
        )
        self.sequence_offset = np.append(self.sequence_offset, len(self.data_list))

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]
        data_list = []
        for split in self.split:
            data_list += glob.glob(os.path.join(self.data_root, split, "*", "*"))
        return data_list

    @staticmethod
    def align_pose(coord, pose, target_pose):
        coord = np.hstack((coord, np.ones_like(coord[:, :1])))
        pose_align = np.matmul(np.linalg.inv(target_pose), pose)
        coord = (pose_align @ coord.T).T[:, :3]
        return coord

    def get_single_frame(self, idx):
        return super().get_data(idx)

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            return self.get_single_frame(idx)

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        major_frame = self.get_single_frame(idx)
        name = major_frame.pop("name")
        target_pose = major_frame.pop("pose")
        for key in major_frame.keys():
            major_frame[key] = [major_frame[key]]

        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
            refer_frame = self.get_single_frame(refer_idx)
            refer_frame.pop("name")
            pose = refer_frame.pop("pose")
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
            )
            if not self.reference_label:
                refer_frame["segment"] = (
                    np.ones_like(refer_frame["segment"]) * self.ignore_index
                )

            if self.timing_embedding:
                refer_frame["strength"] = np.hstack(
                    (
                        refer_frame["strength"],
                        np.ones_like(refer_frame["strength"]) * timestamp,
                    )
                )

            for key in major_frame.keys():
                major_frame[key].append(refer_frame[key])
        for key in major_frame.keys():
            major_frame[key] = np.concatenate(major_frame[key], axis=0)
        major_frame["name"] = name
        return major_frame

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        sequence_path, frame_name = os.path.split(file_path)
        sequence_name = os.path.basename(sequence_path)
        data_name = f"{sequence_name}_{frame_name}"
        return data_name
