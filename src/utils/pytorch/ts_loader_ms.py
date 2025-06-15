import numpy as np
import pandas as pd
import copy
from mindspore import Tensor, ops, nn
from ..pytorch.ts_dataset_ms import TimeSeriesDatasetMS  # 你自己准备的 MindSpore 版 Dataset
from collections import defaultdict


class TimeSeriesLoaderMS(object):
    def __init__(self,
                 ts_dataset: TimeSeriesDatasetMS,
                 model: str,
                 offset: int,
                 window_sampling_limit: int,
                 input_size: int,
                 output_size: int,
                 idx_to_sample_freq: int,
                 batch_size: int,
                 is_train_loader: bool,
                 shuffle: bool):

        self.model = model
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.idx_to_sample_freq = idx_to_sample_freq
        self.offset = offset
        self.ts_dataset = ts_dataset
        self.t_cols = self.ts_dataset.t_cols
        self.is_train_loader = is_train_loader
        self.shuffle = shuffle

        self._create_train_data()

    def _update_sampling_windows_idxs(self):
        outsample_condition = ops.ReduceSum(keep_dims=False)(
            self.ts_windows[:, self.t_cols.index('outsample_mask'), -self.output_size:], axis=1
        )
        insample_condition = ops.ReduceSum(keep_dims=False)(
            self.ts_windows[:, self.t_cols.index('insample_mask'), :self.input_size], axis=1
        )
        mask = outsample_condition * insample_condition > 0
        sampling_idx = np.nonzero(mask.asnumpy())[0].tolist()
        return sampling_idx

    def _create_windows_tensor(self):
        tensor_np_raw, right_padding, train_mask_np = self.ts_dataset.get_filtered_ts_tensor(
            offset=self.offset,
            output_size=self.output_size,
            window_sampling_limit=self.window_sampling_limit
        )
        tensor = Tensor(tensor_np_raw.astype(np.float32))
        train_mask = Tensor(train_mask_np.astype(np.float32))

        if not self.is_train_loader:
            train_mask = 1 - train_mask
        mask_expanded = ops.tile(ops.expand_dims(train_mask, 0), (tensor.shape[0], 1))
        outsample_idx = self.t_cols.index('outsample_mask')
        tensor[:, outsample_idx, :] = tensor[:, outsample_idx, :] * mask_expanded

        pad_op = nn.Pad(paddings=((0, 0), (0, 0), (self.input_size, right_padding)))
        tensor = pad_op(tensor)

        y_idx = self.t_cols.index('y')
        tensor[:, y_idx, -self.output_size:] = 0
        tensor[:, outsample_idx, -self.output_size:] = 0

        # === 替换 Unfold 和 permute 的部分 ===
        tensor_np = tensor.asnumpy()
        windows = []
        for series in tensor_np:
            n_channels, T = series.shape
            series_windows = []
            for i in range(0, T - self.input_size - self.output_size + 1, self.idx_to_sample_freq):
                window = series[:, i:i + self.input_size + self.output_size]
                series_windows.append(window)
            if series_windows:
                series_windows = np.stack(series_windows, axis=0)
                windows.append(series_windows)

        assert windows, "No valid windows created. Please check input_size, output_size, idx_to_sample_freq"
        windows = np.concatenate(windows, axis=0)
        windows = Tensor(windows.astype(np.float32))

        repeat_factor = int(len(windows) / self.ts_dataset.n_series)
        s_matrix = np.tile(self.ts_dataset.s_matrix, (repeat_factor, 1))

        return windows, Tensor(s_matrix.astype(np.float32))

    def __len__(self):
        return len(self.windows_sampling_idx)

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.permutation(self.windows_sampling_idx)
        else:
            sample_idxs = self.windows_sampling_idx

        assert len(sample_idxs) > 0, 'Check the data as sample_idxs are empty'
        n_batches = int(np.ceil(len(sample_idxs) / self.batch_size))

        for idx in range(n_batches):
            ws_idxs = sample_idxs[(idx * self.batch_size): (idx + 1) * self.batch_size]
            yield self.__get_item__(index=ws_idxs)

    def __get_item__(self, index):
        if self.model == 'nbeats':
            return self._nbeats_batch(index)
        elif self.model == 'esrnn':
            assert False, 'hacer esrnn'
        else:
            assert False, 'error'

    def _nbeats_batch(self, index):
        if isinstance(index, np.ndarray):
            index = index.tolist()
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        y_idx = self.t_cols.index('y')
        insample_mask_idx = self.t_cols.index('insample_mask')
        outsample_mask_idx = self.t_cols.index('outsample_mask')

        insample_y = windows[:, y_idx, :self.input_size]
        insample_x = windows[:, (y_idx + 1):insample_mask_idx, :self.input_size]
        insample_mask = windows[:, insample_mask_idx, :self.input_size]

        outsample_y = windows[:, y_idx, self.input_size:]
        outsample_x = windows[:, (y_idx + 1):insample_mask_idx, self.input_size:]
        outsample_mask = windows[:, outsample_mask_idx, self.input_size:]

        batch = {
            's_matrix': s_matrix,
            'insample_y': insample_y,
            'insample_x': insample_x,
            'insample_mask': insample_mask,
            'outsample_y': outsample_y,
            'outsample_x': outsample_x,
            'outsample_mask': outsample_mask
        }
        return batch

    def _create_train_data(self):
        self.ts_windows, self.s_matrix = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()

    def update_offset(self, offset):
        if offset == self.offset:
            return
        self.offset = offset
        self._create_train_data()

    def get_meta_data_col(self, col):
        return self.ts_dataset.get_meta_data_col(col)

    def get_n_variables(self):
        return self.ts_dataset.n_x, self.ts_dataset.n_s

    def get_n_series(self):
        return self.ts_dataset.n_series

    def get_max_len(self):
        return self.ts_dataset.max_len

    def get_n_channels(self):
        return self.ts_dataset.n_channels

    def get_X_cols(self):
        return self.ts_dataset.X_cols

    def get_frequency(self):
        return self.ts_dataset.frequency
