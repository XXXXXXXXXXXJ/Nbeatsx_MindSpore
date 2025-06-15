# nbeats_mindspore.py
import copy
import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import Tensor, context
from mindspore.common.initializer import Normal, initializer, HeUniform
from .tcn_mindspore import TemporalConvNet

def filter_input_vars(insample_y, insample_x_t, outsample_x_t, t_cols, include_var_dict):
    # assert include_var_dict['week_day'] == [-1], f"include_var_dict['week_day'] must be [-1], got {include_var_dict['week_day']}"

    outsample_y = ops.Zeros()((insample_y.shape[0], 1, outsample_x_t.shape[2]), mindspore.float32)
    # print("outsample_y shape:", outsample_y.shape)
    insample_y_aux = insample_y.expand_dims(1)
    # print("insample_y_aux shape:", insample_y_aux.shape)

    insample_x_t_aux = ops.Concat(1)((insample_y_aux, insample_x_t))
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("insample_x_t_aux shape:", insample_x_t_aux.shape)
    outsample_x_t_aux = ops.Concat(1)((outsample_y, outsample_x_t))
    # print("outsample_x_t_aux shape:", outsample_x_t_aux.shape)

    x_t = ops.Concat(-1)((insample_x_t_aux, outsample_x_t_aux))
    batch_size, n_channels, input_size = x_t.shape

    assert input_size == 168 + 24, f"input_size {input_size} not 168+24"

    x_t = x_t.view(batch_size, n_channels, 8, 24)
    # print("x_t reshaped shape:", x_t.shape)

    input_vars = []
    day_var = None
    for var in include_var_dict:
        if len(include_var_dict[var]) > 0:
            t_col_idx = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            # print(f"Processing var: {var}, t_col_idx: {t_col_idx}, t_col_filter: {t_col_filter}")
            if var != 'week_day':
                input_vars.append(x_t[:, t_col_idx, t_col_filter, :])
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = ops.Concat(1)(input_vars)
    x_t_filter = x_t_filter.view(batch_size, -1)

    if len(include_var_dict['week_day']) > 0 and day_var is not None:
        x_t_filter = ops.Concat(1)((x_t_filter, day_var))

    # print("Returning x_t_filter shape:", x_t_filter.shape)
    # print("=== Exiting filter_input_vars ===")
    return x_t_filter


class Softplus(nn.Cell):
  def __init__(self):
    super().__init__()
    self.log = ops.Log()
    self.exp = ops.Exp()

  def construct(self, x):
    return self.log(1. + self.exp(x))


class _StaticFeaturesEncoder(nn.Cell):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        self.encoder = nn.SequentialCell([
            nn.Dropout(0.5),
            nn.Dense(in_features, out_features),
            nn.ReLU()
        ])

    def construct(self, x):
        return self.encoder(x)


class NBeatsBlock(nn.Cell):
  def __init__(self, x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, theta_n_dim, basis,
               n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, activation):
    super().__init__()

    if x_s_n_inputs == 0:
      x_s_n_hidden = 0
    theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

    self.x_s_n_inputs = x_s_n_inputs
    self.x_s_n_hidden = x_s_n_hidden
    self.include_var_dict = include_var_dict
    self.t_cols = t_cols
    self.batch_normalization = batch_normalization
    self.dropout_prob = dropout_prob


    self.activations = {
      'relu': nn.ReLU(), 'softplus': Softplus(), 'tanh': nn.Tanh(), 'selu': nn.SeLU(),
      'lrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(), 'sigmoid': nn.Sigmoid()
    }

    layers = []
    for i in range(n_layers):
      # layers.append(nn.Dense(theta_n_hidden[i], theta_n_hidden[i + 1]))
      layers.append(nn.Dense(theta_n_hidden[i], theta_n_hidden[i + 1]))
      layers.append(self.activations[activation])
      if self.batch_normalization:
        layers.append(nn.BatchNorm1d(theta_n_hidden[i + 1]))
      if self.dropout_prob>0:
        layers.append(nn.Dropout(keep_prob=1.0 - self.dropout_prob))

    layers.append(nn.Dense(theta_n_hidden[-1], theta_n_dim))
    if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
      self.static_encoder = _StaticFeaturesEncoder(x_s_n_inputs, x_s_n_hidden)

    self.layers = nn.SequentialCell(layers)
    self.basis = basis

  def construct(self, insample_y, insample_x_t, outsample_x_t, x_s):
    # print("=== Entering NBeatsBlock construct ===")
    # print("insample_y shape before filter:", insample_y.shape)
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)
    # print("x_s shape:", x_s.shape)

    local_include_var_dict = copy.deepcopy(self.include_var_dict)
    # print("local_include_var_dict:", local_include_var_dict)


    if local_include_var_dict is not None:
      insample_y = filter_input_vars(insample_y, insample_x_t, outsample_x_t, self.t_cols, local_include_var_dict)
      # print("insample_y shape after filter_input_vars:", insample_y.shape)

    if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
      x_s = self.static_encoder(x_s)
      # print("x_s shape after static_encoder:", x_s.shape)
      insample_y = ops.Concat(1)((insample_y, x_s))
      # print("insample_y shape after concat with x_s:", insample_y.shape)

    theta = self.layers(insample_y)
    # print("theta shape:", theta.shape)
    backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)
    # print("backcast shape:", backcast.shape)
    # print("forecast shape:", forecast.shape)

    # print("=== Exiting NBeatsBlock construct ===")

    return backcast, forecast


class NBeats(nn.Cell):
  def __init__(self, blocks):
    super().__init__()
    self.blocks = blocks

  def construct(self, insample_y, insample_x_t, insample_mask, outsample_x_t, x_s, return_decomposition=False):
    # print("=== Entering NBeats construct ===")
    # print("Initial insample_y shape:", insample_y.shape)
    # print("Initial insample_x_t shape:", insample_x_t.shape)
    # print("Initial insample_mask shape:", insample_mask.shape)
    # print("Initial outsample_x_t shape:", outsample_x_t.shape)
    # print("Initial x_s shape:", x_s.shape)
    reverse = ops.ReverseV2(axis=[-1])
    residuals = reverse(insample_y)
    insample_x_t = reverse(insample_x_t)
    insample_mask = reverse(insample_mask)
    # print("residuals shape after reverse:", residuals.shape)
    # print("insample_x_t shape after reverse:", insample_x_t.shape)
    # print("insample_mask shape after reverse:", insample_mask.shape)

    forecast = insample_y[:, -1:]
    # print("Initial forecast shape:", forecast.shape)

    block_forecasts = []
    for i, block in enumerate(self.blocks):
      # print(f"Processing block {i}")
      # print(f"residuals shape before block {i}:", residuals.shape)
      backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t, x_s)
      # print(f"Block {i} backcast shape:", backcast.shape)
      # print(f"Block {i} block_forecast shape:", block_forecast.shape)
      residuals = (residuals - backcast) * insample_mask
      # print(f"residuals shape after block {i}:", residuals.shape)
      forecast = forecast + block_forecast
      # print(f"forecast shape after block {i}:", forecast.shape)
      block_forecasts.append(block_forecast)

    block_forecasts = ops.Stack()(block_forecasts)
    block_forecasts = ops.Transpose()(block_forecasts, (1, 0, 2))
    # print("Final block_forecasts shape:", block_forecasts.shape)
    # print("Final forecast shape:", forecast.shape)

    # print("=== Exiting NBeats construct ===")
    if return_decomposition:
      return forecast, block_forecasts
    else:
      return forecast

  def decomposed_prediction(self, insample_y, insample_x_t, insample_mask, outsample_x_t):
    reverse = ops.ReverseV2(axis=[-1])
    residuals = reverse(insample_y)
    insample_x_t = reverse(insample_x_t)
    insample_mask = reverse(insample_mask)

    forecast = insample_y[:, -1:]  # Level with Naive1
    forecast_components = []
    for i, block in enumerate(self.blocks):
      backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
      residuals = (residuals - backcast) * insample_mask
      forecast = forecast + block_forecast
      forecast_components.append(block_forecast)
    return forecast, forecast_components

class IdentityBasis(nn.Cell):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        # print("IdentityBasis initialized: backcast_size =", backcast_size, "forecast_size =", forecast_size)

    def construct(self, theta, insample_x_t, outsample_x_t):
        # print("=== Entering IdentityBasis construct ===")
        # print("theta shape:", theta.shape)
        # print("insample_x_t shape:", insample_x_t.shape)
        # print("outsample_x_t shape:", outsample_x_t.shape)
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        # print("backcast shape:", backcast.shape)
        # print("forecast shape:", forecast.shape)
        # print("=== Exiting IdentityBasis construct ===")
        return backcast, forecast


class TrendBasis(nn.Cell):
    def __init__(self, degree_of_polynomial, backcast_size, forecast_size):
        super().__init__()
        p = degree_of_polynomial + 1
        b = np.concatenate([np.power(np.arange(backcast_size, dtype=np.float32) / backcast_size, i)[None, :] for i in range(p)])
        f = np.concatenate([np.power(np.arange(forecast_size, dtype=np.float32) / forecast_size, i)[None, :] for i in range(p)])
        self.backcast_basis = Parameter(Tensor(b, mindspore.float32), requires_grad=False)
        self.forecast_basis = Parameter(Tensor(f, mindspore.float32), requires_grad=False)
        # print("backcast_basis shape:", self.backcast_basis.shape)
        # print("forecast_basis shape:", self.forecast_basis.shape)

    def construct(self, theta, insample_x_t, outsample_x_t):
        # print("theta shape in TrendBasis:", theta.shape)
        cut_point = self.forecast_basis.shape[0]
        # print("cut_point:", cut_point)
        # print("theta[:, :cut_point] shape:", theta[:, :cut_point].shape)
        # print("theta[:, cut_point:] shape:", theta[:, cut_point:].shape)
        forecast_theta = theta[:, :cut_point].expand_dims(1)
        forecast = ops.BatchMatMul()(forecast_theta, self.forecast_basis).squeeze(1)
        backcast_theta = theta[:, cut_point:].expand_dims(1)
        backcast = ops.BatchMatMul()(backcast_theta, self.backcast_basis).squeeze(1)
        # print("forecast shape after matmul:", forecast.shape)
        # print("backcast shape after matmul:", backcast.shape)
        return backcast, forecast


class SeasonalityBasis(nn.Cell):
  def __init__(self, harmonics, backcast_size, forecast_size):
    super().__init__()
    frequency = np.append(np.zeros(1, dtype=np.float32),
                          np.arange(harmonics, harmonics / 2 * forecast_size,
                                    dtype=np.float32) / harmonics)[None, :]

    backcast_grid = -2 * np.pi * (
      np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
    forecast_grid = 2 * np.pi * (
      np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

    backcast_cos_template = np.transpose(np.cos(backcast_grid))
    backcast_sin_template = np.transpose(np.sin(backcast_grid))
    backcast_template = np.concatenate([backcast_cos_template, backcast_sin_template], axis=0)

    forecast_cos_template = np.transpose(np.cos(forecast_grid))
    forecast_sin_template = np.transpose(np.sin(forecast_grid))
    forecast_template = np.concatenate([forecast_cos_template, forecast_sin_template], axis=0)

    self.backcast_basis = Parameter(Tensor(backcast_template, mindspore.float32), requires_grad=False)
    self.forecast_basis = Parameter(Tensor(forecast_template, mindspore.float32), requires_grad=False)
    # print("SeasonalityBasis initialized: harmonics =", harmonics)
    # print("backcast_basis shape:", self.backcast_basis.shape)
    # print("forecast_basis shape:", self.forecast_basis.shape)

  def construct(self, theta, insample_x_t, outsample_x_t):
    # print("=== Entering SeasonalityBasis construct ===")
    # print("theta shape:", theta.shape)
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)
    cut_point = self.forecast_basis.shape[0]
    # print("cut_point:", cut_point)
    # print("theta[:, :cut_point] shape:", theta[:, :cut_point].shape)
    forecast_theta = theta[:, :cut_point].expand_dims(1)
    forecast = ops.BatchMatMul()(forecast_theta, self.forecast_basis).squeeze(1)
    backcast_theta = theta[:, cut_point:].expand_dims(1)
    backcast = ops.BatchMatMul()(backcast_theta, self.backcast_basis).squeeze(1)
    # print("backcast shape after matmul:", backcast.shape)
    # print("forecast shape after matmul:", forecast.shape)
    # print("=== Exiting SeasonalityBasis construct ===")
    return backcast, forecast


class ExogenousBasisInterpretable(nn.Cell):
  def __init__(self):
    super().__init__()
    # print("ExogenousBasisInterpretable initialized")

  def construct(self, theta, insample_x_t, outsample_x_t):
    # print("=== Entering ExogenousBasisInterpretable construct ===")
    # print("theta shape:", theta.shape)
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)
    backcast_basis = insample_x_t
    forecast_basis = outsample_x_t
    # print("backcast_basis shape:", backcast_basis.shape)
    # print("forecast_basis shape:", forecast_basis.shape)
    cut_point = forecast_basis.shape[1]
    # print("cut_point:", cut_point)
    # print("theta[:, :cut_point] shape:", theta[:, :cut_point].shape)
    # print("theta[:, cut_point:] shape:", theta[:, cut_point:].shape)
    forecast_theta = theta[:, :cut_point].expand_dims(1)
    forecast = ops.BatchMatMul()(forecast_theta, forecast_basis).squeeze(1)
    backcast_theta = theta[:, cut_point:].expand_dims(1)
    backcast = ops.BatchMatMul()(backcast_theta, backcast_basis).squeeze(1)
    # print("backcast shape after matmul:", backcast.shape)
    # print("forecast shape after matmul:", forecast.shape)
    # print("=== Exiting ExogenousBasisInterpretable construct ===")
    return backcast, forecast


class Chomp1d(nn.Cell):
  def __init__(self, chomp_size):
    super().__init__()
    self.chomp_size = chomp_size
    # print("Chomp1d initialized: chomp_size =", chomp_size)

  def construct(self, x):
    # print("=== Entering Chomp1d construct ===")
    # print("input shape:", x.shape)
    # print("input min/max:", x.min().asnumpy(), x.max().asnumpy())
    output = x[:, :, :-self.chomp_size]
    # print("output shape:", output.shape)
    # print("output min/max:", output.min().asnumpy(), output.max().asnumpy())
    # print("=== Exiting Chomp1d construct ===")
    return output


class ExogenousBasisWavenet(nn.Cell):
  def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
    super(ExogenousBasisWavenet, self).__init__()
    # 初始化权重参数
    self.weight = Parameter(initializer(
      HeUniform(math.sqrt(0.5)),
      (1, in_features, 1),
      mindspore.float32),
      name='weight')
    # print("ExogenousBasisWavenet initialized: out_features =", out_features, "in_features =", in_features,
    #       "num_levels =", num_levels, "kernel_size =", kernel_size, "dropout_prob =", dropout_prob)
    # print("weight shape:", self.weight.shape)
    # 计算第一层的padding
    padding = (kernel_size - 1) * (2 ** 0)

    # 创建输入层
    input_layer = nn.SequentialCell([nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                              kernel_size=kernel_size, pad_mode='pad', padding=padding, dilation=2 ** 0),
                                    Chomp1d(padding),
                                    nn.ReLU(),
                                    nn.Dropout(keep_prob=1.0 - dropout_prob)
                                  ])
    # print("input_layer created with padding =", padding)
    # 创建卷积层
    conv_layers = []
    for i in range(1, num_levels):
      dilation = 2 ** i
      padding = (kernel_size - 1) * dilation
      conv_layers.extend([
        nn.Conv1d(in_channels=out_features, out_channels=out_features,
                  kernel_size=kernel_size, pad_mode='pad', padding=padding, dilation=dilation),
        Chomp1d(padding),
        nn.ReLU(),
        # Dropout层只在需要时添加
        nn.Dropout(keep_prob=1.0 - dropout_prob) if dropout_prob > 0 else nn.Identity()
      ])
      # print(f"conv_layer {i} created with dilation =", dilation, "padding =", padding)

    # 合并所有层
    self.wavenet = nn.SequentialCell([input_layer] + conv_layers)
    # print("wavenet layers:", [layer for layer in self.wavenet.cells()])

  def transform(self, insample_x_t, outsample_x_t):
    # print("=== Entering ExogenousBasisWavenet transform ===")
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("insample_x_t min/max:", insample_x_t.min().asnumpy(), insample_x_t.max().asnumpy())
    # print("outsample_x_t shape:", outsample_x_t.shape)
    # print("outsample_x_t min/max:", outsample_x_t.min().asnumpy(), outsample_x_t.max().asnumpy())
    # 获取输入尺寸
    input_size = insample_x_t.shape[2]
    # print("input_size:", input_size)

    # 在时间维度(第2维)拼接张量
    x_t = ops.cat([insample_x_t, outsample_x_t], axis=2)
    # print("x_t shape after cat:", x_t.shape)
    # print("weight shape:", self.weight.shape)

    # 应用权重 - 元素级乘法，在 batch 和时间维度广播
    x_t = x_t * self.weight
    # print("x_t shape after weight multiplication:", x_t.shape)
    # print("x_t min/max after weight:", x_t.min().asnumpy(), x_t.max().asnumpy())
    # print("wavenet input shape:", x_t.shape)

    # 通过 WaveNet 网络
    x_t = self.wavenet(x_t)
    # print("x_t shape after wavenet:", x_t.shape)
    # print("x_t min/max after wavenet:", x_t.min().asnumpy(), x_t.max().asnumpy())

    # 分割结果
    backcast_basis = x_t[:, :, :input_size]
    forecast_basis = x_t[:, :, input_size:]
    # print("backcast_basis shape:", backcast_basis.shape)
    # print("forecast_basis shape:", forecast_basis.shape)
    # print("backcast_basis min/max:", backcast_basis.min().asnumpy(), backcast_basis.max().asnumpy())
    # print("forecast_basis min/max:", forecast_basis.min().asnumpy(), forecast_basis.max().asnumpy())
    # print("=== Exiting ExogenousBasisWavenet transform ===")

    return backcast_basis, forecast_basis

  def construct(self, theta, insample_x_t, outsample_x_t):
    # print("=== Entering ExogenousBasisWavenet construct ===")
    # print("theta shape:", theta.shape)
    # print("theta min/max:", theta.min().asnumpy(), theta.max().asnumpy())
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)

    backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
    # print("backcast_basis shape after transform:", backcast_basis.shape)
    # print("forecast_basis shape after transform:", forecast_basis.shape)
    cut_point = forecast_basis.shape[1]
    # print("cut_point:", cut_point)
    # print("theta[:, :cut_point] shape:", theta[:, :cut_point].shape)
    # print("theta[:, cut_point:] shape:", theta[:, cut_point:].shape)
    # print("theta_forecast min/max:", theta[:, :cut_point].min().asnumpy(), theta[:, :cut_point].max().asnumpy())
    # print("theta_backcast min/max:", theta[:, cut_point:].min().asnumpy(), theta[:, cut_point:].max().asnumpy())
    # print("Before forecast matmul: theta[:, :cut_point] shape =", theta[:, :cut_point].shape, "forecast_basis shape =",
    #       forecast_basis.shape)
    forecast_theta = theta[:, :cut_point].expand_dims(1)
    forecast = ops.BatchMatMul()(forecast_theta, forecast_basis).squeeze(1)
    # print("After forecast matmul shape:", forecast.shape)
    # print("forecast min/max:", forecast.min().asnumpy(), forecast.max().asnumpy())
    # print("Before backcast matmul: theta[:, cut_point:] shape =", theta[:, cut_point:].shape, "backcast_basis shape =",
    #       backcast_basis.shape)
    backcast_theta = theta[:, cut_point:].expand_dims(1)
    backcast = ops.BatchMatMul()(backcast_theta, backcast_basis).squeeze(1)
    # print("After backcast matmul shape:", backcast.shape)
    # print("backcast min/max:", backcast.min().asnumpy(), backcast.max().asnumpy())
    # print("=== Exiting ExogenousBasisWavenet construct ===")
    return backcast, forecast




class ExogenousBasisTCN(nn.Cell):
  def __init__(self, out_features, in_features, num_levels=4, kernel_size=2, dropout_prob=0):
    super().__init__()
    n_channels = num_levels * [out_features]
    self.tcn = TemporalConvNet(in_features, n_channels, kernel_size, dropout_prob)
    # print("ExogenousBasisTCN initialized: out_features =", out_features, "in_features =", in_features, "num_levels =",
    #       num_levels, "kernel_size =", kernel_size, "dropout_prob =", dropout_prob)
    # print("tcn layers:", [layer for layer in self.tcn.cells()])

  def transform(self, insample_x_t, outsample_x_t):
    # print("=== Entering ExogenousBasisTCN transform ===")
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)
    x_t = ops.Concat(2)((insample_x_t, outsample_x_t))
    # print("x_t shape after concat:", x_t.shape)
    x_t = self.tcn(x_t)
    # print("x_t shape after tcn:", x_t.shape)
    input_size = insample_x_t.shape[2]
    # print("input_size:", input_size)
    backcast_basis = x_t[:, :, :input_size]
    forecast_basis = x_t[:, :, input_size:]
    # print("backcast_basis shape:", backcast_basis.shape)
    # print("forecast_basis shape:", forecast_basis.shape)
    # print("=== Exiting ExogenousBasisTCN transform ===")
    return backcast_basis, forecast_basis

  def construct(self, theta, insample_x_t, outsample_x_t):
    # print("=== Entering ExogenousBasisTCN construct ===")
    # print("theta shape:", theta.shape)
    # print("insample_x_t shape:", insample_x_t.shape)
    # print("outsample_x_t shape:", outsample_x_t.shape)
    backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
    # print("backcast_basis shape after transform:", backcast_basis.shape)
    # print("forecast_basis shape after transform:", forecast_basis.shape)
    cut_point = forecast_basis.shape[1]
    # print("cut_point:", cut_point)
    # print("theta[:, :cut_point] shape:", theta[:, :cut_point].shape)
    # print("theta[:, cut_point:] shape:", theta[:, cut_point:].shape)
    forecast_theta = theta[:, :cut_point].expand_dims(1)
    forecast = ops.BatchMatMul()(forecast_theta, forecast_basis).squeeze(1)
    backcast_theta = theta[:, cut_point:].expand_dims(1)
    backcast = ops.BatchMatMul()(backcast_theta, backcast_basis).squeeze(1)
    # print("backcast shape after matmul:", backcast.shape)
    # print("forecast shape after matmul:", forecast.shape)
    # print("=== Exiting ExogenousBasisTCN construct ===")
    return backcast, forecast


if __name__ == '__main__':
  # context.set_context(mode=context.PYNATIVE_MODE)
  mindspore.set_context(device_target="CPU")
  mindspore.set_seed(42)
  np.random.seed(42)

  # 模拟输入数据参数
  batch_size = 2
  backcast_length = 168
  forecast_length = 24
  x_s_dim = 4
  theta_dim = backcast_length + forecast_length
  x_t_channels = 3
  x_s_hidden = 8

  # 输入张量
  insample_y = Tensor(np.random.randn(batch_size, backcast_length), mindspore.float32)
  insample_x_t = Tensor(np.random.randn(batch_size, x_t_channels, backcast_length), mindspore.float32)
  outsample_x_t = Tensor(np.random.randn(batch_size, x_t_channels, forecast_length), mindspore.float32)
  x_s = Tensor(np.random.randn(batch_size, x_s_dim), mindspore.float32)
  insample_mask = Tensor(np.ones_like(insample_y.asnumpy()), mindspore.float32)

  # 网络通用配置
  block_kwargs = dict(
    x_t_n_inputs=backcast_length,
    x_s_n_inputs=x_s_dim,
    x_s_n_hidden=x_s_hidden,
    theta_n_dim=theta_dim,
    n_layers=2,
    theta_n_hidden=[64, 64],
    include_var_dict=None,
    t_cols=[],
    batch_normalization=False,
    dropout_prob=0.0,
    activation='relu'
  )

  # Basis 模块列表
  basis_modules = {
    "IdentityBasis": IdentityBasis(backcast_length, forecast_length),
    "TrendBasis": TrendBasis(degree_of_polynomial=3, backcast_size=backcast_length, forecast_size=forecast_length),
    "SeasonalityBasis": SeasonalityBasis(harmonics=5, backcast_size=backcast_length, forecast_size=forecast_length),
    "ExogenousBasisInterpretable": ExogenousBasisInterpretable(),
    "ExogenousBasisTCN": ExogenousBasisTCN(out_features=4, in_features=x_t_channels, num_levels=2, kernel_size=2),
    "ExogenousBasisWavenet": ExogenousBasisWavenet(out_features=4, in_features=x_t_channels, num_levels=2, kernel_size=3)
  }

  # 遍历测试所有 basis
  for name, basis in basis_modules.items():
    print(f"\n===== Testing {name} =====")

    block = NBeatsBlock(
      basis=basis,
      **block_kwargs
    )

    model = nn.CellList([block])
    nbeats = nn.SequentialCell([block])

    # 前向测试
    nbeats.set_train(False)
    try:
      output = nbeats(insample_y, insample_x_t, outsample_x_t, x_s)
      print(f"Output shape: {output.shape}")
      print(f"Output first 5 values: {output[0, :5]}")
    except Exception as e:
      print(f"[ERROR] Failed for {name}: {e}")

    # 参数统计
    for p in model.get_parameters():
      print(p.name, p.shape)
    total_params = sum([np.prod(p.shape) for p in block.get_parameters()])
    print(f"Parameter count for {name}: {total_params:,}")

    for name, p in block.parameters_and_names():
      print(name, p.shape)
    total_params = sum(np.prod(p.shape) for _, p in block.parameters_and_names() if p.requires_grad)
    print(f"Parameter count for {name}: {total_params:,}")



