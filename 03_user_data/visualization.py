from PIL import Image  
import PIL as plt

def visulaze_augmentations(dataset,idx = 0.num_versions = 8):
    """see what augmentation actully does to your image"""
    fig,axes = plt.subplot(2,3,figsize(12,6))
    axes = axes.flatten()

    for i in range(num_versions):
        img,label = dataset[idx]

        img = denormalize(img)
        axes[i].imshow(img.permute(1,2,0)) #CHW->HWC
        axes[i].set_title(f"Version{i+1}")
        axes[i].axis('off')

    plt.suptitle(f"Same flower (index{idx}),8 different augmentations")
    plt.tight_layout
    plt.show(0)
