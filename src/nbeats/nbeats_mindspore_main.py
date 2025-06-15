# nbeats_mindspore_main.py
import os
import time
import numpy as np
import pandas as pd
import random
import gc
import copy

from collections import defaultdict

from pathlib import Path
from functools import partial

import mindspore as ms
from mindspore.ops import clip_by_global_norm
from mindspore import nn, ops, Tensor, Parameter, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer, HeUniform, HeNormal, XavierUniform, XavierNormal
from .nbeats_mindspore import NBeats, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis,ExogenousBasisInterpretable, ExogenousBasisWavenet, ExogenousBasisTCN
from utils.pytorch.losses_ms import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss, PinballLoss
from utils.numpy.metrics import mae, pinball_loss, mape, smape, mse, rmse
"""
# >>> test init begin
def glorot_uniform_np(shape, gain=1.0, dtype=np.float32):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(dtype)


def init_weights_shared_ms(cell, shared_weights):
  import mindspore.nn as nn
  dense_idx = 0
  for sub in cell.cells():
    if isinstance(sub, nn.Dense):
      if dense_idx >= len(shared_weights):
        raise IndexError(
          f"Too many Dense layers ({dense_idx + 1}) but only {len(shared_weights)} shared weights provided.")
      weight_np, bias_np = shared_weights[dense_idx]
      sub.weight.set_data(Tensor(weight_np, dtype=sub.weight.dtype))
      sub.bias.set_data(Tensor(bias_np, dtype=sub.bias.dtype))
      dense_idx += 1

  assert dense_idx == len(shared_weights), \
    f"Mismatch: expected {len(shared_weights)} Dense layers, but found {dense_idx}"


# <<< test init end
"""

def init_weights(cell, initialization):
    for subcell in cell.cells():
        if isinstance(subcell, nn.Dense):
            if initialization == 'orthogonal':
                pass  # MindSpore 不直接支持 orthogonal
            elif initialization == 'he_uniform':
                subcell.weight.set_data(initializer(HeUniform(), subcell.weight.shape, subcell.weight.dtype))
            elif initialization == 'he_normal':
                subcell.weight.set_data(initializer(HeNormal(), subcell.weight.shape, subcell.weight.dtype))
            elif initialization == 'glorot_uniform':
                subcell.weight.set_data(initializer(XavierUniform(), subcell.weight.shape, subcell.weight.dtype))
            elif initialization == 'glorot_normal':
                subcell.weight.set_data(initializer(XavierNormal(), subcell.weight.shape, subcell.weight.dtype))
            elif initialization == 'lecun_normal':
                pass  # 用默认即可
            else:
                raise ValueError(f"Unknown initialization: {initialization}")


class NbeatsMS:
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self,
                 input_size_multiplier,
                 output_size,
                 shared_weights,
                 activation,
                 initialization,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_hidden,
                 n_harmonics,
                 n_polynomials,
                 exogenous_n_channels,
                 include_var_dict,
                 t_cols,
                 batch_normalization,
                 dropout_prob_theta,
                 dropout_prob_exogenous,
                 x_s_n_hidden,
                 learning_rate,
                 lr_decay,
                 n_lr_decay_steps,
                 weight_decay,
                 l1_theta,
                 n_iterations,
                 early_stopping,
                 loss,
                 loss_hypar,
                 val_loss,
                 random_seed,
                 seasonality,
                 device=None):
      super().__init__()

      if activation == 'selu':
        initialization = 'lecun_normal'

      self.input_size = int(input_size_multiplier * output_size)
      self.output_size = output_size
      self.shared_weights = shared_weights
      self.activation = activation
      self.initialization = initialization
      self.stack_types = stack_types
      self.n_blocks = n_blocks
      self.n_layers = n_layers
      self.n_hidden = n_hidden
      self.n_harmonics = n_harmonics
      self.n_polynomials = n_polynomials
      self.exogenous_n_channels = exogenous_n_channels
      self.include_var_dict = include_var_dict
      self.t_cols = t_cols

      self.batch_normalization = batch_normalization
      self.dropout_prob_theta = dropout_prob_theta
      self.dropout_prob_exogenous = dropout_prob_exogenous
      self.x_s_n_hidden = x_s_n_hidden

      self.learning_rate = learning_rate
      self.lr_decay = lr_decay
      self.n_lr_decay_steps = n_lr_decay_steps
      self.weight_decay = weight_decay
      self.l1_theta = l1_theta
      self.l1_conv = 1e-3  # 固定值，非超参数
      self.n_iterations = n_iterations
      self.early_stopping = early_stopping
      self.loss = loss
      self.loss_hypar = loss_hypar
      self.val_loss = val_loss
      self.random_seed = random_seed

      self.seasonality = seasonality
      if device is None:
        device = 'GPU' if ms.context.get_context('device_target') == 'GPU' else 'CPU'
      self.device = device
      self._is_instantiated = False

    def create_stack(self):
      if self.include_var_dict is not None:
        x_t_n_inputs = self.output_size * int(sum([len(x) for x in self.include_var_dict.values()]))
        if len(self.include_var_dict['week_day']) > 0:
          x_t_n_inputs = x_t_n_inputs - self.output_size + 1
      else:
        x_t_n_inputs = self.input_size

      block_list = []
      self.blocks_regularizer = []  # 重置
      for i, block_type in enumerate(self.stack_types):
        for block_id in range(self.n_blocks[i]):
          # 仅第一个块使用批归一化（如果启用）
          if (len(block_list) == 0) and (self.batch_normalization):
            batch_normalization_block = True
          else:
            batch_normalization_block = False

          # 初始化为0（表示不需要卷积正则化）
          self.blocks_regularizer.append(0)

          # 共享权重处理
          if self.shared_weights and block_id > 0:
            nbeats_block = block_list[-1]
          else:
            if block_type == 'seasonality':
              basis = SeasonalityBasis(self.n_harmonics, self.input_size, self.output_size)
              theta_dim = 4 * int(np.ceil(self.n_harmonics / 2 * self.output_size) - (self.n_harmonics - 1))
            elif block_type == 'trend':
              basis = TrendBasis(self.n_polynomials, self.input_size, self.output_size)
              theta_dim = 2 * (self.n_polynomials + 1)
            elif block_type == 'identity':
              basis = IdentityBasis(self.input_size, self.output_size)
              theta_dim = self.input_size + self.output_size
            elif block_type == 'exogenous':
              basis = ExogenousBasisInterpretable()
              theta_dim = 2*self.n_x_t
            elif block_type == 'exogenous_tcn':
              basis = ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t)
              theta_dim = 2 * self.exogenous_n_channels
            elif block_type == 'exogenous_wavenet':
              basis = ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t)
              theta_dim = 2 * self.exogenous_n_channels
              self.blocks_regularizer[-1] = 1  # 标记需要卷积正则化
            else:
              raise ValueError(f"Unsupported block type: {block_type}")

          nbeats_block = NBeatsBlock(
            x_t_n_inputs=x_t_n_inputs,
            x_s_n_inputs=self.n_x_s,
            x_s_n_hidden=self.x_s_n_hidden,
            theta_n_dim=theta_dim,
            basis=basis,
            n_layers=self.n_layers[i],
            theta_n_hidden=self.n_hidden[i],
            include_var_dict=self.include_var_dict,
            t_cols=self.t_cols,
            batch_normalization=batch_normalization_block,
            dropout_prob=self.dropout_prob_theta,
            activation=self.activation
          )
          """
          # >>> test init begin
          if hasattr(self, "shared_test_init_weights"):
            shared_idx = len(block_list)  # 当前正在构造第几个 block
            if shared_idx < len(self.shared_test_init_weights):
              init_weights_shared_ms(nbeats_block.layers, self.shared_test_init_weights[shared_idx])
          # <<< test init end
          """
          init_function = partial(init_weights, initialization=self.initialization)
          nbeats_block.layers.apply(init_function)

          block_list.append(nbeats_block)

      return block_list

    def __loss_fn(self, loss_name: str):
      def loss(x, loss_hypar, forecast, target, mask):
        if loss_name == 'MAPE':
          return MAPELoss(y=target, y_hat=forecast, mask=mask) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        elif loss_name == 'MASE':
          return MASELoss(y=target, y_hat=forecast, y_insample=x, seasonality=loss_hypar, mask=mask) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        elif loss_name == 'SMAPE':
          return SMAPELoss(y=target, y_hat=forecast, mask=mask) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        elif loss_name == 'MSE':
          return MSELoss(y=target, y_hat=forecast, mask=mask) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        elif loss_name == 'MAE':
          return MAELoss(y=target, y_hat=forecast, mask=mask) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        elif loss_name == 'PINBALL':
          return PinballLoss(y=target, y_hat=forecast, mask=mask, tau=loss_hypar) + \
            self.loss_l1_conv_layers() + self.loss_l1_theta()
        else:
          raise Exception(f'Unknown loss function: {loss_name}')

      return loss

    def __val_loss_fn(self, loss_name='MAE'):
      def loss(forecast, target, weights):
        if loss_name == 'MAPE':
          return mape(y=target, y_hat=forecast, weights=weights)
        elif loss_name == 'SMAPE':
          return smape(y=target, y_hat=forecast, weights=weights)
        elif loss_name == 'MSE':
          return mse(y=target, y_hat=forecast, weights=weights)
        elif loss_name == 'RMSE':
          return rmse(y=target, y_hat=forecast, weights=weights)
        elif loss_name == 'MAE':
          return mae(y=target, y_hat=forecast, weights=weights)
        elif loss_name == 'PINBALL':
          return pinball_loss(y=target, y_hat=forecast, weights=weights, tau=0.5)
        else:
          raise Exception(f'Unknown loss function: {loss_name}')

      return loss

    def loss_l1_conv_layers(self):
      loss_l1 = 0
      for i, indicator in enumerate(self.blocks_regularizer):
        if indicator:
          for name, param in self.model.blocks[i].parameters_and_names():
            if 'wavenet' in name and 'weight' in name:
              loss_l1 += self.l1_conv * ops.abs(param).sum()
      return loss_l1

    def loss_l1_theta(self):
      loss_l1 = 0
      for block in self.model.blocks:
        for _, cell in block.cells_and_names():
          if isinstance(cell, nn.Dense):
            loss_l1 += self.l1_theta * ops.abs(cell.weight).sum()
      return loss_l1

    def to_tensor(self, x):
      return Tensor(x, ms.float32)

    def fit(self, train_ts_loader, val_ts_loader=None, n_iterations=None, verbose=True, eval_steps=1):
      assert (self.input_size) == train_ts_loader.input_size, \
        f'model input_size {self.input_size} data input_size {train_ts_loader.input_size}'

      # Random Seeds (model initialization)
      ms.set_seed(self.random_seed)
      np.random.seed(self.random_seed)
      random.seed(self.random_seed)

      # Attributes of ts_dataset
      self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()

      # Instantiate model
      if not self._is_instantiated:
        block_list = self.create_stack()
        self.model = NBeats(nn.CellList(block_list))
        self._is_instantiated = True

      # Overwrite n_iterations and train datasets
      if n_iterations is None:
        n_iterations = self.n_iterations

      lr_decay_steps = max(n_iterations // self.n_lr_decay_steps, 1)

      params = self.model.trainable_params()
      optimizer = nn.Adam(params, learning_rate=self.learning_rate, weight_decay=self.weight_decay)
      training_loss_fn = self.__loss_fn(self.loss)
      validation_loss_fn = self.__val_loss_fn(self.val_loss)

      # 梯度裁剪
      # grad_clip = nn.ClipByGlobalNorm(1.0)

      # 训练轨迹记录
      print('\n' + '=' * 30 + ' Start fitting ' + '=' * 30)
      start_time = time.time()
      self.trajectories = {'iteration': [], 'train_loss': [], 'val_loss': []}
      self.final_insample_loss = None
      self.final_outsample_loss = None

      # 训练循环变量
      best_val_loss = float('inf')
      best_params = None
      best_insample_loss = None
      early_stopping_counter = 0
      iteration = 0
      epoch = 0
      break_flag = False

      def forward_fn(insample_y, insample_x, insample_mask, outsample_x, s_matrix, outsample_y, outsample_mask):
        forecast = self.model(insample_y, insample_x, insample_mask, outsample_x, s_matrix)
        loss = training_loss_fn(x=insample_y, loss_hypar=self.loss_hypar, forecast=forecast,
                                target=outsample_y, mask=outsample_mask)
        return loss

      grad_fn = ms.grad(forward_fn, grad_position=None, weights=params)
      # 训练循环
      while iteration < n_iterations and not break_flag:
        epoch += 1
        self.model.set_train()

        for batch in train_ts_loader:
          if iteration >= n_iterations or break_flag:
            break

          # 准备数据
          insample_y = self.to_tensor(batch['insample_y'])
          insample_x = self.to_tensor(batch['insample_x'])
          insample_mask = self.to_tensor(batch['insample_mask'])
          outsample_x = self.to_tensor(batch['outsample_x'])
          outsample_y = self.to_tensor(batch['outsample_y'])
          outsample_mask = self.to_tensor(batch['outsample_mask'])
          s_matrix = self.to_tensor(batch['s_matrix'])

          loss = forward_fn(insample_y, insample_x, insample_mask, outsample_x, s_matrix, outsample_y,
                            outsample_mask)
          grads = grad_fn(insample_y, insample_x, insample_mask, outsample_x, s_matrix, outsample_y,
                          outsample_mask)
          grads = clip_by_global_norm(grads, clip_norm=1.0)
          optimizer(grads)

          # 更新学习率 (模拟StepLR)
          if iteration % lr_decay_steps == 0 and iteration > 0:
            optimizer.learning_rate.set_data(optimizer.learning_rate * self.lr_decay)

          # 记录和评估
          if iteration % eval_steps == 0:
            train_loss_np = loss.asnumpy()
            self.trajectories['iteration'].append(iteration)
            self.trajectories['train_loss'].append(train_loss_np)

            log_str = f'Step: {iteration}, Time: {time.time() - start_time:.2f}s, Insample {self.loss}: {train_loss_np:.5f}'

            # 验证集评估
            if val_ts_loader is not None:
              val_loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                         validation_loss_fn=validation_loss_fn)
              self.trajectories['val_loss'].append(val_loss)
              log_str += f', Outsample {self.val_loss}: {val_loss:.5f}'

              # 早停检查
              if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = copy.deepcopy(self.model.parameters_dict())
                best_insample_loss = train_loss_np
                early_stopping_counter = 0
              else:
                early_stopping_counter += 1

              if self.early_stopping and early_stopping_counter >= self.early_stopping:
                if verbose:
                  print('\n')
                  print(19 * '-', ' Stopped training by early stopping', 19 * '-')
                break_flag = True

            if verbose:
              print(log_str)

          iteration += 1

      # 恢复最佳参数
      if best_params is not None:
        load_param_into_net(self.model, best_params)
        self.final_insample_loss = best_insample_loss
        self.final_outsample_loss = best_val_loss
      else:
        self.final_insample_loss = loss.asnumpy() if iteration > 0 else float('nan')

      # 最终评估
      if verbose:
        log_str = f'Step: {min(iteration, n_iterations)}, Time: {time.time() - start_time:.2f}s, Insample {self.loss}: {self.final_insample_loss:.5f}'
        if val_ts_loader is not None:
          self.final_outsample_loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                                      validation_loss_fn=validation_loss_fn)
          log_str += f', Outsample {self.val_loss}: {self.final_outsample_loss:.5f}'
        print(log_str)

      print('=' * 30 + '  End fitting  ' + '=' * 30 + '\n')
      return self

    def predict(self, ts_loader, X_test=None, return_decomposition=False):
      """预测整个数据集"""
      self.model.set_train(False)
      forecasts = []
      block_forecasts = []
      outsample_ys = []
      outsample_masks = []

      for batch in ts_loader:
        # 准备数据
        insample_y = self.to_tensor(batch['insample_y'])
        insample_x = self.to_tensor(batch['insample_x'])
        insample_mask = self.to_tensor(batch['insample_mask'])
        outsample_x = self.to_tensor(batch['outsample_x'])
        s_matrix = self.to_tensor(batch['s_matrix'])

        # 获取预测
        forecast, block_forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                                      insample_mask=insample_mask, outsample_x_t=outsample_x,
                                                      x_s=s_matrix, return_decomposition=True) # always return decomposition

        # 收集结果
        forecasts.append(forecast.asnumpy())
        block_forecasts.append(block_forecast.asnumpy())
        outsample_ys.append(batch['outsample_y'])
        outsample_masks.append(batch['outsample_mask'])

      # 拼接结果
      forecasts = np.vstack(forecasts)
      block_forecasts = np.vstack(block_forecasts)
      outsample_ys = np.vstack(outsample_ys)
      outsample_masks = np.vstack(outsample_masks)

      if return_decomposition:
        return outsample_ys, forecasts, block_forecasts, outsample_masks
      else:
        return outsample_ys, forecasts, outsample_masks

    def evaluate_performance(self, ts_loader, validation_loss_fn):
      """评估模型性能"""
      # 获取预测结果（不返回分解）
      target, forecast, mask = self.predict(ts_loader, return_decomposition=False)

      # 计算损失
      loss = validation_loss_fn(forecast, target, mask)
      return loss

    def save(self, model_dir, model_id):
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
      model_path = os.path.join(model_dir, f"model_{model_id}.ckpt")
      save_checkpoint(self.model, model_path)
      print(f'Saved model to: {model_path}')

    def load(self, model_dir, model_id):
      model_path = os.path.join(model_dir, f"model_{model_id}.ckpt")
      param_dict = load_checkpoint(model_path)
      load_param_into_net(self.model, param_dict)
      print(f'Loaded model from: {model_path}')
