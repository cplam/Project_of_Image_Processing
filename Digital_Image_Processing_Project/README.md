# rey_code

## Project Overview
This project implements an image segmentation pipeline for skin regions. The main methods include:
- Skin-threshold segmentation in the YCbCr color space
- Otsu global thresholding
- Morphological operations (closing and opening) for mask refinement

The program processes input images one by one and outputs intermediate and final segmentation results for analysis.

## Requirements
- Python 3.8 or above
- numpy
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
This repository does not include the `dataset` files. Please download them separately.

- Dataset link: `https://hkustconnect-my.sharepoint.com/:u:/g/personal/cyinad_connect_ust_hk/IQCNow8SZvUETaov02APLKZsAZeSQhh_NCS48VENpPt5b-A?e=petvyg`


```text
Project/
  dataset/
    00008.png
    00066.png
    ...
  segmentationFinal.py
```

## Input Files
```
Please refer to file: input_images
```

## Output Files
For each input image, the script generates:
```
member 1: output_member1
member 2: output_member2
member 5: GUI save anywhere in computers
```