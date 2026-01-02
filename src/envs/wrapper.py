"""
Wrapper for the DonkeyCar environment

Image Processing:
- ROI Cropping
- Resizing: 80x 80 or 160 x 80
- Grayscale & Binary Thresholding
- Normalization: scale RGB 0 ~ 1
- Frame Stacking : ( 80, 80, 4 ) or ( 160, 80, 4 )
"""