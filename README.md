# VAE

---

### Model

```python
class Encoder(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.linear_mean = nn.Linear(2048, output_dim)
        self.linear_logvar = nn.Linear(2048, output_dim)
        
    def forward(self, inputs):
        # (batch, 1, 28, 28)
        x = self.conv2d_1(inputs).relu()
        # (batch, 32, 14, 14)
        x = self.conv2d_2(x).relu()
        # (batch, 64, 7, 7)
        x = self.conv2d_3(x).relu()
        # (batch, 128, 4, 4)
        x = x.reshape(-1, 2048).relu()
        # (batch, 2048)
        z_mean = self.linear_mean(x)
        z_logvar = self.linear_logvar(x) # 분산에 로그
        return z_mean, z_logvar
    
    
class Decoder(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2048)
        self.convt_2d_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2)
        self.convt_2d_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2)
        self.convt_2d_3 = nn.ConvTranspose2d(32, 1, kernel_size=4, padding=1, stride=2)
        # self.convt_2d_4 = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, inputs):
        # (batch, 2)
        x = self.linear(inputs).relu()
        # (batch, 2048)
        x = x.reshape(-1, 128, 4, 4)
        # (batch, 128, 4, 4)
        x = self.convt_2d_1(x).relu()
        # (batch, 64, 7, 7)
        x = self.convt_2d_2(x).relu()
        # (batch, 32, 14, 14)
        x = self.convt_2d_3(x).sigmoid()
        return x
        
        
class VAE(nn.Module):
    def __init__(self, emb_dim=2):
        super().__init__()
        self.encoder = Encoder(emb_dim)
        self.decoder = Decoder(emb_dim)
        
    def forward(self, inputs):
        z_mean, z_logvar = self.encoder(inputs)
        z = z_mean + (z_logvar/2).exp() * torch.randn_like(z_logvar) * self.training
        
        outputs = self.decoder(z)
        return outputs, z_mean, z_logvar
    
```

### Loss

```python
def vae_loss(x_pred, x_true, z_mean, z_logvar):
    recon_loss = F.binary_cross_entropy(x_pred, x_true)
    kl_loss = 0.5 * torch.mean(z_mean**2 + z_logvar.exp() - z_logvar - 1)
    return recon_loss, kl_loss
```

---

### Loss History

![image](https://github.com/tetrapod0/VAE/assets/48349693/b14d3054-b28a-4a4e-8b86-f8865433abf5)

### Reconstructed image

![image](https://github.com/tetrapod0/VAE/assets/48349693/092506d2-6ac6-4a4d-a2a7-cb30fea27456)

### distribution of latent space

![image](https://github.com/tetrapod0/VAE/assets/48349693/ae5978c0-afff-41c8-b100-f583571cae37)

### Generated image

![image](https://github.com/tetrapod0/VAE/assets/48349693/2eb6a21a-6ed5-4e45-b39e-892cc1129da3)











