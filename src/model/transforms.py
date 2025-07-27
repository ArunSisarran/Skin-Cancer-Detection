from torchvision import transforms


def get_transforms(phase='train', img_size=224):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.2,   
                contrast=0.3,     
                saturation=0.25,  
                hue=0.1          
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_advanced_transforms(phase='train', img_size=224, augmentation_level='medium'):
    """Wrapper function for compatibility"""
    return get_transforms(phase, img_size)


def get_melanoma_specific_augmentations(img_size=224):
    """Melanoma-specific augmentations using torchvision"""
    return get_transforms('train', img_size)