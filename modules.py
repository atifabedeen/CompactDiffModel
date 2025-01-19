import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# Removed the SelfAttention class because nn.MultiheadAttention is not quantization-friendly.


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.relu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.double_conv,
            [['0', '1', '2'], ['3', '4']],
            inplace=True
        )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def fuse_model(self):
        self.maxpool_conv[1].fuse_model()
        self.maxpool_conv[2].fuse_model()


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # Removed SelfAttention modules
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # Removed SelfAttention
        x3 = self.down2(x2, t)
        # Removed SelfAttention
        x4 = self.down3(x3, t)
        # Removed SelfAttention

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        # Removed SelfAttention
        x = self.up2(x, x2, t)
        # Removed SelfAttention
        x = self.up3(x, x1, t)
        # Removed SelfAttention
        output = self.outc(x)
        return output

    def fuse_model(self):
        self.inc.fuse_model()
        self.down1.fuse_model()
        self.down2.fuse_model()
        self.down3.fuse_model()
        self.bot1.fuse_model()
        self.bot2.fuse_model()
        self.bot3.fuse_model()
        self.up1.fuse_model()
        self.up2.fuse_model()
        self.up3.fuse_model()


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # Removed SelfAttention modules
        self.down2 = Down(128, 256)
        # Removed SelfAttention modules
        self.down3 = Down(256, 256)
        # Removed SelfAttention modules

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # Removed SelfAttention modules
        self.up2 = Up(256, 64)
        # Removed SelfAttention modules
        self.up3 = Up(128, 64)
        # Removed SelfAttention modules
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # Removed SelfAttention
        x3 = self.down2(x2, t)
        # Removed SelfAttention
        x4 = self.down3(x3, t)
        # Removed SelfAttention

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        # Removed SelfAttention
        x = self.up2(x, x2, t)
        # Removed SelfAttention
        x = self.up3(x, x1, t)
        # Removed SelfAttention
        output = self.outc(x)
        return output

    def fuse_model(self):
        self.inc.fuse_model()
        self.down1.fuse_model()
        self.down2.fuse_model()
        self.down3.fuse_model()
        self.bot1.fuse_model()
        self.bot2.fuse_model()
        self.bot3.fuse_model()
        self.up1.fuse_model()
        self.up2.fuse_model()
        self.up3.fuse_model()


# Initialize the quantization configuration
import torch.quantization

torch.backends.quantized.engine = 'fbgemm'  # Use 'qnnpack' for ARM CPUs

# net = UNet(device="cpu")
net = UNet_conditional(num_classes=10, device="cpu")
net.eval()
# Fuse the model
net.fuse_model()

# Set the quantization configuration for QAT
net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
net.train()
# Prepare the model for QAT
torch.quantization.prepare_qat(net, inplace=True)
print(net)
print(sum([p.numel() for p in net.parameters()]))
x = torch.randn(3, 3, 64, 64)
t = x.new_tensor([500] * x.shape[0]).long()
y = x.new_tensor([1] * x.shape[0]).long()
print(net(x, t, y).shape)

# Convert the model to a quantized version after training
# net.eval()
# net_int8 = torch.quantization.convert(net)

# Test inference with the quantized model
# output = net_int8(x, t, y)
# print(output.shape)
