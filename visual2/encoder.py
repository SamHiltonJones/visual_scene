import torch.nn as nn
import torch
import time
# from sourceMobileNet import mobilenet_v3_small
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.models.layers import trunc_normal_, DropPath

class MobileNetV3Encoder(nn.Module):
    def __init__(self, depth_in):
        super().__init__()

        model = mobilenet_v3_small(weights=None, norm_layer=None)

        # Adjust the first convolutional layer
        # The original first conv layer has 3 input channels. We change it to 4.
        #
        #model.features = remove_batch_norm(model.features)

        self.model = model.features
        print(self.model)

    def forward(self, x):
        return self.model(x)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func, activation=nn.ReLU, reduced_activation=False):
        super().__init__()

        def identity(p): return p
        self.activation1 = activation()
        self.activation2 = activation() if not reduced_activation else identity

        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    #@torch.autocast('cuda')
    def forward(self, x):
        x_ = self.conv_0(self.activation1(x))
        x_ = self.conv_1(self.activation2(x_))
        return x + x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func, activation=nn.ReLU, strided_conv=False, patchify=False, reduced_activation=False):
        super().__init__()
        self.strided_conv = strided_conv
        self.patchify = patchify

        if self.strided_conv:
            self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=2, padding=1)
        elif self.patchify:
            self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=4, stride=4)
        else:
            self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
            self.max_pool = nn.MaxPool2d(3, 2, padding=1)

        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation, reduced_activation=reduced_activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation, reduced_activation=reduced_activation)

    #@torch.autocast('cuda')
    def forward(self, x):
        x = self.conv(x)

        if not self.strided_conv and not self.patchify:
            x = self.max_pool(x)

        x = self.residual_0(x)

        x = self.residual_1(x)

        return x

class ImpalaEncoderMod(nn.Module):
    def __init__(self, in_depth, model_size=2, norm_func=torch.nn.utils.parametrizations.spectral_norm, activation=nn.ReLU,
                 reduced_activation=False, patchify=False):
        super().__init__()
        self.conv = nn.Sequential(
            ImpalaCNNBlock(in_depth, int(16*model_size), norm_func=norm_func, activation=activation, patchify=patchify, reduced_activation=reduced_activation),
            ImpalaCNNBlock(int(16*model_size), int(32*model_size), norm_func=norm_func, activation=activation, reduced_activation=reduced_activation),
            ImpalaCNNBlock(int(32*model_size), int(32*model_size), norm_func=norm_func, activation=activation, reduced_activation=reduced_activation),
        )

        if not reduced_activation:
            self.conv.append(activation())

        if not patchify:
            self.conv.append(torch.nn.AdaptiveMaxPool2d((6, 6)))

    def forward(self, x):
        return self.conv(x)


class ImpalaTrueRes(nn.Module):
    def __init__(self, in_depth, model_size=2, norm_func=torch.nn.utils.parametrizations.spectral_norm, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            ImpalaCNNResBlock(in_depth, int(16*model_size), norm_func=norm_func, activation=activation),
            ImpalaCNNResBlock(int(16*model_size), int(32*model_size), norm_func=norm_func, activation=activation),
            ImpalaCNNResBlock(int(32*model_size), int(32*model_size), norm_func=norm_func, activation=activation),
            activation(),
            torch.nn.AdaptiveMaxPool2d((6, 6))
        )

    def forward(self, x):
        return self.conv(x)

class Impala3DBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func, activation=nn.ReLU):
        super().__init__()

        self.conv3d = nn.Conv3d(
            in_channels=1,           # Input channels
            out_channels=depth_out,  # Number of output channels (filters)
            kernel_size=(2, 3, 3),   # Kernel size
            stride=(1, 2, 2),        # Stride
            padding=(0, 1, 1)        # Padding for height and width dimensions
        )

        # make it happen
        self.channel_mixer = nn.Conv3d(
            in_channels=depth_out,           # Input channels
            out_channels=depth_out,  # Number of output channels (filters)
            kernel_size=(3, 1, 1),   # Kernel size
            stride=(1, 1, 1),        # Stride
            padding=(0, 0, 0),        # Padding for height and width dimensions
            groups=32
        )

        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)

    #@torch.autocast('cuda')
    def forward(self, x):
        x = self.conv3d(x)

        x = self.channel_mixer(x)

        x = x.squeeze()

        x = self.residual_0(x)

        x = self.residual_1(x)

        return x


class Impala3D(nn.Module):
    def __init__(self, depth_in, norm_func, activation=nn.ReLU, model_size=2):
        super().__init__()
        self.conv = nn.Sequential(
            Impala3DBlock(4, 16 * model_size, norm_func=norm_func, activation=activation),
            ImpalaCNNBlock(16 * model_size, 32 * model_size, norm_func=norm_func, activation=activation),
            ImpalaCNNBlock(32 * model_size, 32 * model_size, norm_func=norm_func, activation=activation),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.conv(x)


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=4, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(2): # this was 4. I'm using a halved version with 4 block (2, and 2)
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        #self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(2):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.maxpool(x)
        return x # self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_RL(**kwargs):
    model = ConvNeXtV2(depths=[2, 1], dims=[40, 80], **kwargs)
    return model


if __name__ == "__main__":
    device = torch.device('cuda:0')

    """encoder = ImpalaEncoder(4).to(device)

    start = time.time()
    for i in range(5000):
        state = torch.rand((1, 4, 84, 84), dtype=torch.float32, device=device)

        result = encoder(state)

    print(result.shape)
    print(time.time() - start)

    raise Exception("stop")"""
    encoder = ImpalaEncoderMod(4).to(device)
    from torchsummary import summary

    summary(encoder, (4, 84, 84))

    start = time.time()
    for i in range(5000):
        state = torch.rand((1, 4, 84, 84), dtype=torch.float32, device=device)

        result = encoder(state)

    print(result.shape)
    print(time.time() - start)






