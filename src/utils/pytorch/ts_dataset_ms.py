
import numpy as np
import pandas as pd
from collections import defaultdict

class TimeSeriesDatasetMS:
    """
    MindSpore版时间序列数据集构建类，对齐 PyTorch 版本结构
    """
    def __init__(self, Y_df, X_df=None, S_df=None, f_cols=None, ts_train_mask=None):
        assert isinstance(Y_df, pd.DataFrame)
        assert all([(col in Y_df.columns) for col in ['unique_id', 'ds', 'y']])
        if X_df is not None:
            assert isinstance(X_df, pd.DataFrame)
            assert all([(col in X_df.columns) for col in ['unique_id', 'ds']])

        self.f_cols = f_cols  # 必须放在 _df_to_lists 之前

        print("处理数据帧 ...")
        ts_data, s_data, self.meta_data, self.t_cols, self.X_cols = self._df_to_lists(Y_df, S_df, X_df)

        self.n_series = len(ts_data)
        self.max_len = max(len(ts['y']) for ts in ts_data)
        self.n_channels = len(self.t_cols)
        self.frequency = pd.infer_freq(Y_df.head()['ds'])

        self.n_x = 0 if X_df is None else len(self.X_cols)
        self.n_s = 0 if S_df is None else S_df.shape[1] - 1

        print("构建时间序列张量 ...")
        self.ts_tensor, self.s_matrix, self.len_series = self._create_tensor(ts_data, s_data)

        if ts_train_mask is None:
            ts_train_mask = np.ones(self.max_len)
        assert len(ts_train_mask) == self.max_len

        self._declare_outsample_train_mask(ts_train_mask)

    def _df_to_lists(self, Y_df, S_df, X_df):
        unique_ids = Y_df['unique_id'].unique()
        X_cols = [col for col in X_df.columns if col not in ['unique_id', 'ds']] if X_df is not None else []
        S_cols = [col for col in S_df.columns if col != 'unique_id'] if S_df is not None else []

        ts_data, s_data, meta_data = [], [], []

        for u_id in unique_ids:
            serie_df = Y_df[Y_df['unique_id'] == u_id]
            ts_data_i = {'y': serie_df['y'].values}

            for X_col in X_cols:
                ts_data_i[X_col] = X_df[X_df['unique_id'] == u_id][X_col].values
            ts_data.append(ts_data_i)

            s_data_i = defaultdict(list)
            for S_col in S_cols:
                s_data_i[S_col] = S_df[S_df['unique_id'] == u_id][S_col].values
            s_data.append(s_data_i)

            meta_data.append({'unique_id': u_id, 'last_ds': serie_df['ds'].max()})

        # 修复点：排除 f_cols 中的列，避免进入 t_cols
        effective_X_cols = [col for col in X_cols if col not in (self.f_cols or [])]
        t_cols = ['y'] + effective_X_cols + ['insample_mask', 'outsample_mask']

        return ts_data, s_data, meta_data, t_cols, X_cols

    def _create_tensor(self, ts_data, s_data):
        s_matrix = np.zeros((self.n_series, self.n_s))
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len))
        len_series = []

        for idx in range(self.n_series):
            ts_idx = np.array([ts_data[idx]['y']] + [ts_data[idx][col] for col in self.X_cols if col in self.t_cols])
            ts_len = ts_idx.shape[1]

            ts_tensor[idx, :self.t_cols.index('insample_mask'), -ts_len:] = ts_idx
            ts_tensor[idx, self.t_cols.index('insample_mask'), -ts_len:] = 1
            ts_tensor[idx, self.t_cols.index('outsample_mask'), -ts_len:] = 1

            if self.n_s > 0:
                s_matrix[idx, :] = [v[0] if isinstance(v, (list, np.ndarray)) else v for v in s_data[idx].values()]

            len_series.append(ts_len)

        return ts_tensor, s_matrix, np.array(len_series)

    def _declare_outsample_train_mask(self, ts_train_mask):
        self.ts_train_mask = ts_train_mask

    def get_meta_data_col(self, col):
        return [x[col] for x in self.meta_data]

    def get_filtered_ts_tensor(self, offset, output_size, window_sampling_limit, ts_idxs=None):
        last_outsample_ds = self.max_len - offset + output_size
        first_ds = max(last_outsample_ds - window_sampling_limit - output_size, 0)

        if ts_idxs is None:
            filtered_ts_tensor = self.ts_tensor[:, :, first_ds:last_outsample_ds]
        else:
            filtered_ts_tensor = self.ts_tensor[ts_idxs, :, first_ds:last_outsample_ds]

        right_padding = max(last_outsample_ds - self.max_len, 0)
        ts_train_mask = self.ts_train_mask[first_ds:last_outsample_ds]

        assert np.sum(np.isnan(filtered_ts_tensor)) < 1.0, "存在 NaN 值"
        return filtered_ts_tensor, right_padding, ts_train_mask

    def get_f_idxs(self, cols):
        assert all(col in self.f_cols for col in cols), f"{cols} 中存在未在 f_cols 声明的列"
        return [self.X_cols.index(col) for col in cols]
