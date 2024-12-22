# CIFAR10 Custom CNN Implementation

This project implements a custom CNN architecture for the CIFAR10 dataset with specific architectural constraints and requirements.

## Model Architecture

The model follows a C1C2C3C40 architecture with the following specifications:
- No MaxPooling (uses strided convolutions instead)
- Uses Depthwise Separable Convolution
- Uses Dilated Convolution
- Global Average Pooling
- Parameters < 200k

### Key Features:
- Channel progression: 3→16→32→48→80→96→112→120
- Receptive Field > 44
- Uses efficient depthwise separable convolutions
- Implements dilated convolutions for increased receptive field
- Strided convolution in C4 block instead of MaxPooling

## Data Augmentation

Uses Albumentations library with the following transformations:
1. Horizontal Flip (p=0.5)
2. ShiftScaleRotate
   - shift_limit: 0.1
   - scale_limit: 0.1
   - rotate_limit: 15
3. CoarseDropout
   - max_holes: 1
   - max_height: 16px
   - max_width: 16px
   - min_holes: 1
   - min_height: 16px
   - min_width: 16px
   - fill_value: dataset mean

## Requirements

```bash
pip install -r requirements.txt
```


## Usage

To train the model:

```bash
python train.py
```


## Model Summary

The model uses:
- Depthwise Separable Convolutions for efficiency
- Dilated Convolutions for increased receptive field
- Strided Convolutions instead of MaxPooling
- Global Average Pooling
- Total parameters: < 200k

## Training Details

- Optimizer: Adam
- Learning Rate Scheduler: OneCycleLR
- Batch Size: 128
- Target Accuracy: 85%

## Results

The model achieves the following results:
- Training Accuracy: 98.5%
- Validation Accuracy: 85%
- Training Loss: 0.01
- Validation Loss: 0.12

## Conclusion

This implementation demonstrates a custom CNN architecture that adheres to the specified constraints and achieves high accuracy on the CIFAR10 dataset.
