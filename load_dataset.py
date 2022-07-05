from utils import ext_transforms as et
from datasets import Monumai


def get_dataset(opts):
    """
    Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtResize(512),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = et.ExtCompose([
        et.ExtResize((512, 512)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dst = Monumai(root=opts.data_root, split='train', transform=train_transform)
    val_dst = Monumai(root=opts.data_root, split='test', transform=val_transform)
    test_dst = Monumai(root=opts.data_root, split='val', transform=val_transform)

    return train_dst, val_dst, test_dst
