import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, initializer
import numpy as np


class WeightNormConv1d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, weight_init=Normal(0.01)):
        super(WeightNormConv1d, self).__init__()

        self.kernel_size_val = kernel_size
        self.stride_val = stride
        self.padding_val = padding
        self.dilation_val = dilation
        self.has_bias = bias

        v_shape = (out_channels, in_channels, kernel_size)
        self.v = Parameter(initializer(weight_init, v_shape, mindspore.float32), name='weight_v')

        # Weight norm scalar g (1D tensor with shape [out_channels])
        self.g = Parameter(
            Tensor(np.linalg.norm(self.v.asnumpy().reshape(out_channels, -1), axis=1).astype(np.float32)),
            name='weight_g'
        )

        # Optional bias
        if self.has_bias:
            self.bias = Parameter(initializer('zeros', [out_channels]), name='bias')
        else:
            self.bias = None

        self.norm_op = ops.L2Normalize(axis=1)
        self.pad_op = ops.Pad(((0, 0), (0, 0), (self.padding_val, self.padding_val)))
        self.conv1d = ops.conv1d

    def construct(self, x):
        v_flat = self.v.view(self.v.shape[0], -1)
        norm_v_flat = self.norm_op(v_flat)
        norm_v = norm_v_flat.view(self.v.shape)
        weight = self.g.view(-1, 1, 1) * norm_v

        x_padded = self.pad_op(x)
        out = self.conv1d(x_padded, weight, stride=self.stride_val,
                          padding=0, dilation=self.dilation_val, groups=1)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out


class Chomp1d(nn.Cell):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def construct(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = WeightNormConv1d(n_inputs, n_outputs, kernel_size, stride,
                                      padding, dilation, bias=True)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(1.0 - dropout)

        self.conv2 = WeightNormConv1d(n_outputs, n_outputs, kernel_size, stride,
                                      padding, dilation, bias=True)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(1.0 - dropout)

        # Downsample should not have bias to match PyTorch
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = WeightNormConv1d(n_inputs, n_outputs, kernel_size=1,
                                               stride=1, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU()
        self.add = ops.Add()

    def construct(self, x):
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.add(out, res))


class TemporalConvNet(nn.Cell):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))
        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


if __name__ == '__main__':
    # 设置种子确保初始化一致
    mindspore.set_seed(42)
    np.random.seed(42)

    # 构建模型
    model = TemporalConvNet(
        num_inputs=3,
        num_channels=[64, 128, 64],
        kernel_size=3,
        dropout=0.0  # 关闭 dropout 方便验证
    )

    # 打印模型结构
    print("MindSpore Model:")
    print(model)

    # 统计参数
    print("\nDetailed MindSpore Parameters:")
    total_params = 0
    for p in model.get_parameters():
        num = p.size
        print(f"  {p.name}: {p.shape}, elements: {num}")
        total_params += num

    print(f"\nDetailed calculated total parameters: {total_params:,}")
    print(f"\nTotal parameters: {total_params:,}")

    # 测试前向传播
    x = Tensor(np.random.randn(1, 3, 100), mindspore.float32)
    model.set_train(False)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output first 5 values: {output[0, 0, :5]}")
