import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
    zero=nn.ZeroPad2d,
    reflect=nn.ReflectionPad2d,
    replicate=nn.ReplicationPad2d,
)

conv_dict = dict(
    conv2d=nn.Conv2d,
    deconv2d=nn.ConvTranspose2d,
)

norm_dict = dict(
    none=lambda x: lambda x: x,
    spectral=lambda x: lambda x: x,
    batch=nn.BatchNorm2d,
    instance=nn.InstanceNorm2d,
    layer=LayerNorm,
)

activ_dict = dict(
    none=lambda: lambda x: x,
    relu=lambda: nn.ReLU(inplace=True),
    lrelu=lambda: nn.LeakyReLU(0.2, inplace=True),
    prelu=lambda: nn.PReLU(),
    selu=lambda: nn.SELU(inplace=True),
    tanh=lambda: nn.Tanh(),
)


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class DeconvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', **conv_opts):
        super(DeconvolutionBlock, self).__init__()

        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ=activ, pad=pad)]
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ='none', pad=pad)]
        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x) + x


class Encoder(nn.Module):
    def __init__(self, input_channels=1):
        super(Encoder, self).__init__()

        self.conv_block_1 = ConvolutionBlock(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1, norm='instance', activ='relu', pad='zero')
        self.residual_block_1 = ResidualBlock(channels=32, pad='zero', norm='instance', activ='relu')

        self.conv_block_2 = ConvolutionBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, norm='instance', activ='relu', pad='zero')
        self.residual_block_2 = ResidualBlock(channels=64, pad='zero', norm='instance', activ='relu')

        self.conv_block_3 = ConvolutionBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, norm='instance', activ='relu', pad='zero')
        self.residual_block_3 = ResidualBlock(channels=128, pad='zero', norm='instance', activ='relu')

        self.conv_block_4 = ConvolutionBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, norm='instance', activ='relu', pad='zero')
        self.residual_block_end = ResidualBlock(channels=256, pad='zero', norm='instance', activ='relu')

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.conv_block_4(x)
        x = self.residual_block_end(x)
        return x


class Decoder(nn.Module):
    def __init__(self, fuse=False, output_channels=1):
        super(Decoder, self).__init__()
        self.fuse = fuse

        input_channels = 512 if self.fuse else 256

        self.residual_block_start = ResidualBlock(channels=input_channels, pad='zero', norm='instance', activ='relu')
        self.conv_block_1 = DeconvolutionBlock(in_channels=input_channels, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, norm='instance', activ='relu', conv='deconv2d')
        self.residual_block_1 = ResidualBlock(channels=128, pad='zero', norm='instance', activ='relu')

        self.conv_block_2 = DeconvolutionBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, norm='instance', activ='relu', conv='deconv2d')
        self.residual_block_2 = ResidualBlock(channels=64, pad='zero', norm='instance', activ='relu')

        self.conv_block_3 = DeconvolutionBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, norm='instance', activ='relu', conv='deconv2d')
        self.residual_block_3 = ResidualBlock(channels=32, pad='zero', norm='instance', activ='relu')

        self.conv_block_4 = DeconvolutionBlock(in_channels=32, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, norm='instance', activ='relu', conv='deconv2d')

    def forward(self, x1, x2=None):
        if self.fuse:
            if x2 is None:
                raise ValueError("x2 must be provided when fuse is set to True")
            x = torch.cat((x1, x2), dim=1)
        else:
            x = x1

        x = self.residual_block_start(x)
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.conv_block_4(x)
        return x


class SpeckleReductionNet(nn.Module):

    def __init__(self, channels=1):
        super(SpeckleReductionNet, self).__init__()
        self.channels = channels

        self.encoder_highRes = Encoder(input_channels=channels)
        self.encoder_lowRes = Encoder(input_channels=channels)
        self.encoder_midRes = Encoder(input_channels=channels)
        self.decoder = Decoder(fuse=False, output_channels=channels)

    def forward(self, hr_img, lr_img, mid_img):
        z_hr = self.encoder_highRes(hr_img)
        z_lr = self.encoder_lowRes(lr_img)
        z_mid = self.encoder_midRes(mid_img)

        i_clean_hr = self.decoder(z_hr)
        i_clean_lr = self.decoder(z_lr)
        i_clean_mid = self.decoder(z_mid)

        return i_clean_hr, i_clean_lr, i_clean_mid
