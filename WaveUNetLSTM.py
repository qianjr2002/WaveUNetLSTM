import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, dilation=dilation, padding=0)

    def forward(self, x):
        # 严格因果填充：只向左侧（过去）填充
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class CausalResBlock(nn.Module):
    """对应图中右侧放大部分：3层 (LeakyReLU -> CausalConv)"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                CausalConv1d(channels, channels, kernel_size=kernel_size)
            ) for _ in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:
            res = x
            x = layer(x)
            x = x + res
        return x

class WaveUNetLSTM(nn.Module):
    def __init__(self, out_channels=1, lookahead=16, sample_shift=1):
        super().__init__()
        in_ch = lookahead // sample_shift + 1 

        # --- Encoder 层级 ---
        # Level 1: 输入 -> 通道调整 -> ResBlock -> 下采样
        self.enc1_pre = nn.Conv1d(in_ch, 32, 1) # 调整通道至32
        self.enc1_res = CausalResBlock(32)
        self.down1 = nn.Conv1d(32, 32, kernel_size=15, stride=4, padding=7)

        # Level 2: 32ch -> 通道调整 -> ResBlock -> 下采样
        self.enc2_pre = nn.Conv1d(32, 64, 1)
        self.enc2_res = CausalResBlock(64)
        self.down2 = nn.Conv1d(64, 64, kernel_size=15, stride=4, padding=7)

        # Level 3: 64ch -> 通道调整 -> ResBlock -> 下采样
        self.enc3_pre = nn.Conv1d(64, 128, 1)
        self.enc3_res = CausalResBlock(128)
        self.down3 = nn.Conv1d(128, 128, kernel_size=15, stride=2, padding=7)

        # --- Bottleneck ---
        self.lstm = nn.LSTM(128, 128, batch_first=True)

        # --- Decoder 层级 ---
        # Level 3: LSTM输出 -> 上采样 -> Concat e3 -> ResBlock
        self.up3 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.dec3_reduce = nn.Conv1d(128 + 128, 128, 1)
        self.dec3_res = CausalResBlock(128)

        # Level 2: y3 -> 上采样 -> Concat e2 -> ResBlock
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2)
        self.dec2_reduce = nn.Conv1d(64 + 64, 64, 1)
        self.dec2_res = CausalResBlock(64)

        # Level 1: y2 -> 上采样 -> Concat e1 -> ResBlock
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2)
        self.dec1_reduce = nn.Conv1d(32 + 32, 32, 1)
        self.dec1_res = CausalResBlock(32)

        self.out_conv = nn.Conv1d(32, out_channels, 1)

    def forward(self, x0):
        # --- Encoder ---
        # Level 1
        e1_feat = self.enc1_res(self.enc1_pre(x0)) # 用于 Skip Conn
        d1 = self.down1(e1_feat)
        
        # Level 2
        e2_feat = self.enc2_res(self.enc2_pre(d1)) # 用于 Skip Conn
        d2 = self.down2(e2_feat)
        
        # Level 3
        e3_feat = self.enc3_res(self.enc3_pre(d2)) # 用于 Skip Conn
        d3 = self.down3(e3_feat)

        # --- LSTM ---
        x_lstm, _ = self.lstm(d3.permute(0, 2, 1))
        x_lstm = x_lstm.permute(0, 2, 1)

        # --- Decoder ---
        # Level 3
        u3 = self.up3(x_lstm)
        u3 = torch.cat([u3[..., :e3_feat.size(-1)], e3_feat], dim=1)
        y3 = self.dec3_res(self.dec3_reduce(u3))

        # Level 2
        u2 = self.up2(y3)
        u2 = torch.cat([u2[..., :e2_feat.size(-1)], e2_feat], dim=1)
        y2 = self.dec2_res(self.dec2_reduce(u2))

        # Level 1
        u1 = self.up1(y2)
        u1 = torch.cat([u1[..., :e1_feat.size(-1)], e1_feat], dim=1)
        y1 = self.dec1_res(self.dec1_reduce(u1))

        return self.out_conv(y1)

if __name__ == "__main__":
    x = torch.randn(1, 17, 32)
    model = WaveUNetLSTM()
    y = model(x)

    print("Input shape :", x.shape) # torch.Size([1, 17, 32])
    print("Output shape:", y.shape) # torch.Size([1, 1, 32])

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (17, 16000))
    print(f'flops:{flops}, params:{params}')
    # flops:2.09 GMac, params:1.3 M
