from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image 


class medical_img_data(Dataset):
    
    def __init__(self, root_dir, data_size):
        
        """
        Args:
            root_dir (string): Directory with all the images.
            data_size = The number of samples 
            
        """
        self.data_size = data_size
        self.train_dir = root_dir + '/PD'
        self.true_dir = root_dir + '/GT'
        self.train_transform = transforms.Compose([transforms.Resize((572,572)),
                                                   transforms.Grayscale(num_output_channels=1),
                                                   transforms.ToTensor()])
        
        
        
        self.true_transform = transforms.Compose([transforms.Resize((572,572)),
                                                  transforms.Grayscale(num_output_channels=1),
                                                  transforms.ToTensor()])

    def __len__(self):
        return self.data_size
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_train_name = os.path.join(self.train_dir,str(idx)+'.png')
        img_true_name = os.path.join(self.true_dir,str(idx)+'.png')
        image_train = Image.open(img_train_name)
        image_true = Image.open(img_true_name)
    
        sample_train = self.train_transform(image_train)
        sample_true = self.true_transform(image_true)
        
        return sample_train,sample_true

