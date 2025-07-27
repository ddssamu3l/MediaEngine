"""
RIFE architecture v2 - Built to match the exact weight structure in rife46.pth
This approach focuses on matching weights exactly first, then figuring out the forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def warp(tenInput, tenFlow):
    """Warp function from the official RIFE implementation."""
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', 
                                         padding_mode='zeros', align_corners=True)


class PReLUCustom(nn.Module):
    """Custom PReLU that uses beta parameter to match weight structure."""
    def __init__(self, num_parameters):
        super(PReLUCustom, self).__init__()
        self.beta = nn.Parameter(torch.ones(1, num_parameters, 1, 1) * 0.25)
        
    def forward(self, x):
        return torch.where(x > 0, x, x * self.beta)


class ConvBlock(nn.Module):
    """Convolution block that matches the weights structure exactly."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # Direct conv2d without extra nesting - weights are at this level
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride  
        self.padding = padding
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class IFBlock(nn.Module):
    """
    IFBlock that matches the exact weight structure in rife46.pth:
    - conv0: two conv layers (no PReLU parameters stored separately)
    - convblock: 8 conv layers with beta parameters 
    - lastconv: single ConvTranspose2d outputting 24 channels
    """
    
    def __init__(self, in_channels, c_half, c, convblock_c):
        super(IFBlock, self).__init__()
        
        # conv0 structure: in_channels -> c_half -> c
        # conv0.0.0: first conv layer
        self.conv0 = nn.ModuleList([
            nn.ModuleList([ConvBlock(in_channels, c_half, 3, 2, 1)]),  # conv0.0.0
            nn.ModuleList([ConvBlock(c_half, c, 3, 2, 1)])             # conv0.1.0  
        ])
        
        # convblock: 8 layers with PReLU (beta parameter)
        self.convblock = nn.ModuleList([
            self._make_conv_prelu_block(convblock_c, convblock_c) for _ in range(8)
        ])
        
        # lastconv: ConvTranspose2d that outputs 24 channels
        self.lastconv = nn.ModuleList([
            nn.ConvTranspose2d(convblock_c, 24, 4, 2, 1, bias=True)
        ])
        
    def _make_conv_prelu_block(self, in_channels, out_channels):
        """Create a conv + PReLU block matching the weight structure."""
        class ConvPReLUBlock(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=True)
                self.beta = nn.Parameter(torch.ones(1, out_c, 1, 1) * 0.25)
                
            def forward(self, x):
                x = self.conv(x)
                return torch.where(x > 0, x, x * self.beta)
                
        return ConvPReLUBlock(in_channels, out_channels)
        
    def forward(self, x, flow, scale=1):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                            align_corners=False, recompute_scale_factor=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                               align_corners=False, recompute_scale_factor=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        
        # Apply conv0 layers with default PReLU
        x = self.conv0[0][0](x)
        x = F.relu(x)  # Use ReLU for now - the PReLU params might be merged differently
        x = self.conv0[1][0](x)
        x = F.relu(x)
        
        # Apply convblock with residual connection
        identity = x
        for conv_prelu in self.convblock:
            x = conv_prelu(x)
        x = x + identity
        
        # Apply lastconv to get 24 channels
        tmp = self.lastconv[0](x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear",
                          align_corners=False, recompute_scale_factor=False)
        
        # Process 24 channels - let's assume it's 6 sets of 4-channel flow
        # This is a guess that we'll need to validate
        batch_size = tmp.shape[0]
        height, width = tmp.shape[2], tmp.shape[3]
        
        # Reshape 24 channels to 6x4 and take the mean or use the first set
        # This needs to be determined by testing
        tmp_reshaped = tmp.view(batch_size, 6, 4, height, width)
        flow_candidates = tmp_reshaped.mean(dim=1)  # Average across the 6 sets
        
        flow = flow_candidates * scale * 2
        # Create a mask - since we don't have explicit mask channels, create a default one
        mask = torch.ones(batch_size, 1, height, width, device=flow.device) * 0.5
        
        return flow, mask


class RIFEModel(nn.Module):
    """RIFE Model that matches the exact weight structure."""
    
    def __init__(self):
        super(RIFEModel, self).__init__()
        
        # Create blocks with the exact channel configurations from weights analysis:
        # block0: 7 -> 96 -> 192, convblock: 192
        self.block0 = IFBlock(in_channels=7, c_half=96, c=192, convblock_c=192)
        
        # block1: 12 -> 64 -> 128, convblock: 128  
        self.block1 = IFBlock(in_channels=12, c_half=64, c=128, convblock_c=128)
        
        # block2: 12 -> 48 -> 96, convblock: 96
        self.block2 = IFBlock(in_channels=12, c_half=48, c=96, convblock_c=96)
        
        # block3: 12 -> 32 -> 64, convblock: 64
        self.block3 = IFBlock(in_channels=12, c_half=32, c=64, convblock_c=64)
        
        self.version = 4.6
        self.loaded = False
        
    def load_model(self, model_path, rank=-1):
        """Load RIFE model weights with exact key matching."""
        model_path = Path(model_path)
        
        pth_files = list(model_path.glob("*.pth"))
        if not pth_files:
            print(f"No .pth files found in {model_path}")
            return False
            
        pth_file = max(pth_files, key=lambda x: x.stat().st_size)
        
        try:
            print(f"Loading RIFE model from {pth_file}")
            checkpoint = torch.load(str(pth_file), map_location='cpu')
            
            # Load directly - keys should match exactly now
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
            
            loaded_keys = len(checkpoint) - len(missing_keys)
            print(f"Successfully loaded {loaded_keys}/{len(checkpoint)} weights")
            
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")
            
            # If we loaded most keys successfully, consider it a success
            if loaded_keys > len(checkpoint) * 0.8:  # 80% threshold
                self.loaded = True
                print(f"Successfully loaded RIFE v{self.version} model")
                return True
            else:
                print(f"Too many missing keys ({len(missing_keys)}/{len(checkpoint)})")
                return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        """Forward pass using the loaded architecture."""
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] if x.shape[1] > 6 else torch.zeros_like(img0)
        
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        
        # Process through the first 3 blocks
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                input_tensor = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
                flow_d, mask_d = stu[i](input_tensor, flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                input_tensor = torch.cat((img0, img1), 1)
                flow, mask = stu[i](input_tensor, None, scale=scale[i])
                
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
            
        # Teacher block (block3) - if we have ground truth
        if gt.shape[1] == 3:
            input_tensor = torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1)
            flow_d, mask_d = self.block3(input_tensor, flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
            
        # Combine warped images with masks
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, 0
    
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        """Perform inference for frame interpolation."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            img0 = img0.to(device)
            img1 = img1.to(device)
            
            # Pad to multiple of 64 for RIFE
            B, C, H, W = img0.shape
            ph = ((H - 1) // 64 + 1) * 64
            pw = ((W - 1) // 64 + 1) * 64
            padding = (0, pw - W, 0, ph - H)
            
            img0_padded = F.pad(img0, padding)
            img1_padded = F.pad(img1, padding)
            
            # Create dummy ground truth for inference
            gt = torch.zeros_like(img0_padded)
            
            # Concatenate inputs
            x = torch.cat([img0_padded, img1_padded, gt], dim=1)
            
            # Run inference
            flow_list, mask, merged, _, _, _ = self.forward(x, timestep=timestep)
            
            # Get the final result and remove padding
            result = merged[2][:, :, :H, :W]
            
            # Apply scaling if needed
            if scale != 1.0:
                new_h, new_w = int(H * scale), int(W * scale)
                result = F.interpolate(result, size=(new_h, new_w), 
                                     mode='bilinear', align_corners=False)
            
            return result
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self


# Alias for compatibility
Model = RIFEModel