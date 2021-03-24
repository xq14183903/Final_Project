import glob
import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
 
pred_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
 
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('best_model.pth')

    tests_path = glob.glob('image/*.png')

    for test_path in tests_path:
  
        save_res_path = test_path.split('.')[0] + '_pred.png'
        
        img = cv2.imread(test_path)
       
        img = pred_transform(img)
       
        inputs = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
        pred = model(inputs)
        
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        
        pred = np.array(pred.data.cpu()[0])[0]

        cv2.imwrite(save_res_path, pred)