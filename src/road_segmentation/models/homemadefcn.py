from torch import nn

class HomemadeFCN(nn.Module):
    def __init__(self):
        super(HomemadeFCN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),

        )

        self.unconv = nn.Sequential(
            nn.ConvTranspose2d(64, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.Sigmoid()
        )

        self.refinement_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
        )

        self.refinement_conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
        )

        self.refinement_conv3 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
        )

        self.refinement_conv4 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
        )

        self.refinement_unconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.refinement_unconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.refinement_unconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.refinement_unconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.unconv(x)
        x = self.refinement_conv1(x)
        x = self.refinement_unconv1(x)
        x = self.refinement_conv2(x)
        x = self.refinement_unconv2(x)
        x = self.refinement_conv3(x)
        x = self.refinement_unconv3(x)
        x = self.refinement_conv4(x)
        x = self.refinement_unconv4(x)
        return x.squeeze()