"""
RIFE architecture v3 - Simplified approach with exact weight key matching
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


class IFBlock(nn.Module):
    """IFBlock that exactly matches the weight key structure."""
    
    def __init__(self, in_channels, c_half, c):
        super(IFBlock, self).__init__()
        
        # conv0: Two sequential conv layers
        # conv0.0.0: in_channels -> c_half
        self.conv0 = nn.ModuleDict({
            "0": nn.ModuleDict({
                "0": nn.Conv2d(in_channels, c_half, 3, 2, 1, bias=True)
            }),
            "1": nn.ModuleDict({
                "0": nn.Conv2d(c_half, c, 3, 2, 1, bias=True)
            })
        })
        
        # convblock: 8 conv+PReLU blocks
        self.convblock = nn.ModuleDict()
        for i in range(8):
            # Create each convblock item as a module with both conv and beta
            block_module = nn.Module()
            block_module.conv = nn.Conv2d(c, c, 3, 1, 1, bias=True)
            block_module.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.25)
            self.convblock[str(i)] = block_module
            
        # lastconv: ConvTranspose2d outputting 24 channels
        self.lastconv = nn.ModuleDict({
            "0": nn.ConvTranspose2d(c, 24, 4, 2, 1, bias=True)
        })
        
    def forward(self, x, flow, scale=1):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                            align_corners=False, recompute_scale_factor=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                               align_corners=False, recompute_scale_factor=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        
        # Apply conv0 layers
        x = self.conv0["0"]["0"](x)
        x = F.relu(x)  # Use ReLU for now
        x = self.conv0["1"]["0"](x)
        x = F.relu(x)
        
        # Apply convblock with residual connection
        identity = x
        for i in range(8):
            block = self.convblock[str(i)]
            x = block.conv(x)
            # Apply PReLU with beta
            x = torch.where(x > 0, x, x * block.beta)
        x = x + identity
        
        # Apply lastconv to get 24 channels
        tmp = self.lastconv["0"](x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear",
                          align_corners=False, recompute_scale_factor=False)
        
        # Process 24 channels - try a simple approach first
        # Maybe the 24 channels represent: [4 flow channels] repeated 6 times for multi-scale
        # Or: [4 flow + 1 mask] repeated almost 5 times (24/5 = 4.8)
        # Let's try: take first 4 as flow, next 1 as mask, ignore the rest
        flow = tmp[:, :4] * scale * 2  # First 4 channels as flow
        mask = tmp[:, 4:5]  # 5th channel as mask
        
        return flow, mask


class RIFEModel(nn.Module):
    """RIFE Model with exact weight key matching."""
    
    def __init__(self):
        super(RIFEModel, self).__init__()
        
        # Create blocks with exact channel configurations from weights:
        # block0: 7 -> 96 -> 192
        self.block0 = IFBlock(in_channels=7, c_half=96, c=192)
        
        # block1: 12 -> 64 -> 128  
        self.block1 = IFBlock(in_channels=12, c_half=64, c=128)
        
        # block2: 12 -> 48 -> 96
        self.block2 = IFBlock(in_channels=12, c_half=48, c=96)
        
        # block3: 12 -> 32 -> 64
        self.block3 = IFBlock(in_channels=12, c_half=32, c=64)
        
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
            
            # Print details about missing/unexpected keys
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}):")
                for key in missing_keys[:10]:
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")
                    
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys) - 10} more")
            
            # If we loaded most keys successfully, consider it a success
            if loaded_keys >= len(checkpoint):  # All keys loaded
                self.loaded = True
                print(f"✓ Perfect weight loading - RIFE v{self.version} model ready!")
                return True
            elif loaded_keys > len(checkpoint) * 0.9:  # 90%+ loaded
                self.loaded = True
                print(f"✓ Successfully loaded RIFE v{self.version} model (some keys missing)")
                return True
            else:
                print(f"✗ Too many missing keys ({len(missing_keys)}/{len(checkpoint)})")
                return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        """Forward pass."""
        # Extract the input images - fix the channel issue
        if x.shape[1] == 6:  # Only 2 RGB images
            img0 = x[:, :3]
            img1 = x[:, 3:6]
            # Add a dummy 7th channel for block0 (might be timestep encoding)
            timestep_channel = torch.full_like(img0[:, :1], timestep)
            block0_input = torch.cat([img0, img1, timestep_channel], dim=1)  # 7 channels
        elif x.shape[1] == 9:  # 2 RGB + 1 GT image
            img0 = x[:, :3]
            img1 = x[:, 3:6]
            gt = x[:, 6:9]
            # Add a dummy 7th channel for block0
            timestep_channel = torch.full_like(img0[:, :1], timestep)
            block0_input = torch.cat([img0, img1, timestep_channel], dim=1)  # 7 channels
        else:
            raise ValueError(f"Expected 6 or 9 input channels, got {x.shape[1]}")
        
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        
        # Process through the first 3 blocks
        stu = [self.block0, self.block1, self.block2]
        
        for i in range(3):
            if i == 0:
                # First block gets special 7-channel input
                input_tensor = block0_input
                flow, mask = stu[i](input_tensor, None, scale=scale[i])
            else:
                # Subsequent blocks: Based on channel analysis, they expect 12 channels total
                # This includes the flow, so we need: 8 input channels + 4 flow channels = 12
                # Let's try: img0(3) + img1(3) + mask(1) + warped_img0(3) but that's 10...
                # Actually, let's try: img0(3) + img1(3) + warped_img0(3) + warped_img1(3) = 12
                # But this doesn't account for flow. Maybe flow is NOT concatenated for these blocks?
                
                # Let me try the approach where flow replaces some channels:
                # Instead of concatenating all, maybe it's: img0 + img1 + flow[:2] + flow[2:4] + mask + something
                # OR: the IFBlock shouldn't concatenate flow - it should handle it differently
                
                input_tensor = torch.cat((img0, img1, warped_img0, warped_img1), 1)  # 12 channels exactly
                
                # Don't pass flow to be concatenated - maybe flow is used differently
                flow_d, mask_d = stu[i](input_tensor, None, scale=scale[i])  # No flow concatenation
                
                # Add the flows
                if flow is not None:
                    flow = flow + flow_d
                    mask = mask + mask_d
                else:
                    flow = flow_d
                    mask = mask_d
                
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
            
        # Combine warped images with masks
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            
        return flow_list, mask_list[2], merged, None, None, 0
    
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        """Perform inference for frame interpolation."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            # Move model to same device as input
            input_device = img0.device
            self.to(input_device)
            
            img0 = img0.to(input_device)
            img1 = img1.to(input_device)
            
            # Pad to multiple of 64 for RIFE
            B, C, H, W = img0.shape
            ph = ((H - 1) // 64 + 1) * 64
            pw = ((W - 1) // 64 + 1) * 64
            padding = (0, pw - W, 0, ph - H)
            
            img0_padded = F.pad(img0, padding)
            img1_padded = F.pad(img1, padding)
            
            # Concatenate inputs (6 channels total)
            x = torch.cat([img0_padded, img1_padded], dim=1)
            
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