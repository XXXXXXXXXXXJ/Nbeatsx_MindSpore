# losses_ms.py
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, ops

def divide_no_nan(a, b):
    """
    辅助函数：防止除以0和处理无穷值
    """
    div = a / b
    div = ops.select(msnp.isnan(div), msnp.zeros_like(div), div)
    div = ops.select(msnp.isinf(div), msnp.zeros_like(div), div)
    return div

#############################################################################
# 预测误差指标损失函数（Forecasting Losses）
#############################################################################

def MAPELoss(y, y_hat, mask=None):
    """
    平均绝对百分比误差（MAPE）
    """
    if mask is None:
        mask = msnp.ones_like(y)
    mask = divide_no_nan(mask, msnp.abs(y))
    mape = msnp.abs(y - y_hat) * mask
    return msnp.mean(mape)

def MSELoss(y, y_hat, mask=None):
    """
    均方误差（MSE）
    """
    if mask is None:
        mask = msnp.ones_like(y)
    mse = (y - y_hat)**2
    mse = mask * mse
    return msnp.mean(mse)

def SMAPELoss(y, y_hat, mask=None):
    """
    对称平均绝对百分比误差（SMAPE）
    """
    if mask is None:
        mask = msnp.ones_like(y)
    delta_y = msnp.abs(y - y_hat)
    scale = msnp.abs(y) + msnp.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = smape * mask
    return 2 * msnp.mean(smape)

def MASELoss(y, y_hat, y_insample, seasonality, mask=None):
    """
    平均绝对缩放误差（MASE）
    """
    if mask is None:
        mask = msnp.ones_like(y)
    delta_y = msnp.abs(y - y_hat)
    scale = msnp.mean(msnp.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    mase = divide_no_nan(delta_y, scale[:, None])
    mase = mase * mask
    return msnp.mean(mase)

def MAELoss(y, y_hat, mask=None):
    """
    平均绝对误差（MAE）
    """
    if mask is None:
        mask = msnp.ones_like(y)
    mae = msnp.abs(y - y_hat) * mask
    return msnp.mean(mae)

def PinballLoss(y, y_hat, mask=None, tau=0.5):
    """
    针对分位回归的 Pinball 损失
    """
    if mask is None:
        mask = msnp.ones_like(y)
    delta_y = y - y_hat
    pinball = msnp.maximum(tau * delta_y, (tau - 1) * delta_y)
    pinball = pinball * mask
    return msnp.mean(pinball)
