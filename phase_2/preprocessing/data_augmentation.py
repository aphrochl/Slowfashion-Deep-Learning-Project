
import torch
from torchvision import transforms

def get_train_transforms():
    """
    Training Data Augmentation Pipeline:
    - Resize: 224x224
    - Random Horizontal Flip (50%)
    - Random Rotation (+/- 15 degrees)
    - Random Affine (Translation, Scaling) to handle positioning
    - Color Jitter (Brightness, Contrast, Saturation)
    - Gaussian Blur (Noise simulation)
    - Normalize (ImageNet stats)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """
    Validation/Test Pipeline:
    - Resize: 224x224
    - Normalize (ImageNet stats)
    (No random augmentation)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    # Quick test to print the composition
    print("Training Transforms:")
    print(get_train_transforms())
    print("\nTesting Transforms:")
    print(get_test_transforms())
