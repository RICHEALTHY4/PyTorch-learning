from torchvision import transforms


#training transforms-with random augumentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RanddomRotation(degrees = 10),
    transforms.ColorJitter(brightness = 0.2),

    #standard preprocessing 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485,0.456,0.406],std = [0.229,0.224,0.225])
])

#validation trnasforms-No augmentation
val_transform = transforms.Compose([
    #only standard preprocessing
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485,0.456,0.406],std = [0.229,0.224,0.225])
])

