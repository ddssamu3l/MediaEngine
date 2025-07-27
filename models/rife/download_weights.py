#!/usr/bin/env python3
"""
Download real RIFE model weights from official sources.
"""

import os
import ssl
import urllib.request
from pathlib import Path

def download_with_fallback(urls, filename):
    """Try downloading from multiple URLs with SSL context handling."""
    # Create SSL context that doesn't verify certificates (for development only)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for url in urls:
        try:
            print(f"Trying to download from: {url}")
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, context=ssl_context) as response:
                with open(filename, 'wb') as f:
                    f.write(response.read())
            
            size = os.path.getsize(filename)
            if size > 10000:  # At least 10KB to verify it's not an error page
                print(f"Successfully downloaded {filename} ({size:,} bytes)")
                return True
            else:
                print(f"Downloaded file too small ({size} bytes), trying next URL...")
                os.remove(filename)
                
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if os.path.exists(filename):
                os.remove(filename)
    
    return False

def main():
    """Download RIFE model weights."""
    base_dir = Path(__file__).parent
    
    # Multiple sources for RIFE weights
    weight_sources = {
        "rife425.pth": [
            "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife425.pth",
            "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife425.pth",
        ],
        "rife46.pth": [
            "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife46.pth",
            "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife46.pth",
        ]
    }
    
    print("Downloading RIFE model weights...")
    
    for filename, urls in weight_sources.items():
        filepath = base_dir / filename
        
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 1000000:  # 1MB threshold for valid weights
                print(f"{filename} already exists ({size:,} bytes)")
                continue
        
        print(f"\nDownloading {filename}...")
        if download_with_fallback(urls, str(filepath)):
            print(f"✓ {filename} downloaded successfully")
        else:
            print(f"✗ Failed to download {filename} from all sources")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()