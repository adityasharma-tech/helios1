import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """U-Net Generator for Radiometric Normalization."""
    def __init__(self):
        super().__init__()
        # Simple U-Net for demonstration
        self.enc1 = self._conv_block(1, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(256, 64) # 128 + 128 (skip)
        
        self.final = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)

class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1), # Input: concatenated real/fake + cond
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        x = torch.cat([img_A, img_B], dim=1)
        return self.model(x)

def train_gan_demo(epochs: int = 2):
    """A dummy training loop to demonstrate the architecture runs."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Initializing GAN models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    
    # Synthetic batch
    B, C, H, W = 2, 1, 256, 256
    real_A = torch.randn(B, C, H, W, device=device) # Source (TMC)
    real_B = torch.randn(B, C, H, W, device=device) # Target (WAC)
    
    opt_G = torch.optim.Adam(gen.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(disc.parameters(), lr=2e-4)
    
    logger.info(f"Starting training on {device}...")
    for epoch in range(epochs):
        # Train Disc
        opt_D.zero_grad()
        fake_B = gen(real_A)
        pred_fake = disc(real_A, fake_B.detach())
        loss_D_fake = torch.mean((pred_fake - 0)**2)
        
        pred_real = disc(real_A, real_B)
        loss_D_real = torch.mean((pred_real - 1)**2)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        opt_D.step()
        
        # Train Gen
        opt_G.zero_grad()
        pred_fake = disc(real_A, fake_B)
        loss_G_GAN = torch.mean((pred_fake - 1)**2)
        loss_G_L1 = torch.nn.functional.l1_loss(fake_B, real_B) * 100
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        opt_G.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")
    
    logger.info("Training complete.")
