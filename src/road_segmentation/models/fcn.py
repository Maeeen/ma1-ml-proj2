from torch import nn

class FCN(nn.Module):
    def __init__(self, layers=4):
        super(FCN, self).__init__()
        
        self.layers = layers

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),

        )

        self.init_unconv = nn.Sequential(
            nn.ConvTranspose2d(64, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.Sigmoid()
        )
        
        self.refinements_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.1),
            )
            for _ in range(layers)
        ])
        
        self.refinements_unconv = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, stride=2, kernel_size=4, padding=1),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
                nn.LeakyReLU(0.1)
            )
            for _ in range(layers)
        ])

    def forward(self, x):
        x = self.conv(x)
        x = self.unconv(x)
        
        for i in range(self.layers):
            x = self.refinements_conv[i](x)
            x = self.refinements_unconv[i](x)

        return x.squeeze()