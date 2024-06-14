# The based unit of graph convolutional networks.  图卷积网络基础单元

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel  时间卷积核大小
        t_stride (int, optional): Stride of the temporal convolution. Default: 1  时间卷积步长
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0  填充，默认为0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1  时间核元素间距，默认为1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 A,
                 t_kernel_size=1,  # 时间核大小
                 t_stride=1,  # 时间步长
                 t_padding=0,  # 填充
                 t_dilation=1,  # 时间核元素间距
                 bias=True):
        super().__init__()
        self.inter_c = 3*out_channels//4

        self.PA = nn.Parameter(torch.from_numpy(np.float32(A)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(np.float32(A)), requires_grad=False)
        # kernel_size:Uni-labeling是1，Distance partitioning是2
        self.kernel_size = 3  # 采用第三种策略，所以kernel_size的取值为3。

        # 空间卷积
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size//4,  # ????
            kernel_size=(t_kernel_size, 1),  # 卷积核的大小（1，1），t_kernel_size的大小视分区策略而定（1，2，3）
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.soft = nn.Softmax(-2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()

        self.tnn = UnfoldTemporalWindows(3, 1, 1)
        self.out_conv = nn.Conv3d(self.inter_c//3, out_channels, kernel_size=(1, 3, 1))
        for i in range(self.kernel_size):
            self.conv_a.append(nn.Conv2d(in_channels, out_channels//4, 1))
            self.conv_b.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels//4, 1))
        # self.conv1 = nn.Conv1d(in_channels=1024,out_channels = 66, kernel_size = 3)
        # self.conv2 = nn.Conv1d(in_channels=1024,out_channels = 66, kernel_size =3)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.relu = nn.ReLU()

    def forward(self, x):
       # assert A.size(0) == self.kernel_size
        N, C, T, V = x.size()

        batch_size = x.size(0)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        # y = None
          # N V V

        # for i in range(self.kernel_size):
        #  A1 = self.conv_a[i](x)
        # A11=A1.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        # A2=A1.view(N, self.inter_c * T, V)
        #  A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
        # A1 = self.soft(torch.matmul(A11, A2) / A1.size(-1))
        # A3 = A1 + A[i]
        # A2 = x.view(N, C * T, V)
        # z = self.conv_d[i](torch.matmul(A2, A3).view(N, C, T, V))
        #  y = z + y if y is not None else z
        x2 = self.tnn(x)
        A1 = self.conv_a[0](x2)
        A11 = A1.permute(0, 3, 1, 2).contiguous().view(N, 3*V, -1)
        A2 = A1.view(N, -1, 3*V)
        A1 = self.soft(torch.matmul(A11, A2) / A1.size(-1))
        A2 = x2.view(N, C * T, 3*V)
        z = self.conv_d[0](torch.matmul(A2, A1).view(N, C, T, 3*V))
        x1 = self.conv(x2)

        n, kc, t, v = x1.size()
        x1 = x1.view(n, self.kernel_size, kc//self.kernel_size, t, v)

        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x1 = x1+z
        x1 = x1.view(N, self.inter_c//3, -1, 3, V)
        y= self.out_conv(x1).squeeze(dim=3)
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=3, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x