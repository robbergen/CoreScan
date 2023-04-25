# CoreScan

This repo classifies mineral types based on the textures in sample core images.

### Data prep
- RGB image path: `/img-rgb-50u`
- Class map label path: `/img-clm-phy`

All images are currently resized to 2048x128 by default.

### Usage
`python main.py`

TODO:
1. Implement parser for inputs
2. Implement option for spectroscopy-based label maps instead of pre-processed class map
