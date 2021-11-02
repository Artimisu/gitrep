import torch
import numpy as np
import ipdb
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image

def load_image(img_path, m, n):
    # img = plt.imread(img_path)
    img = Image.open(img_path)
    random_crop = T.RandomResizedCrop((m, n))
    img = random_crop(img)
    return img

def image_to_tensor(img):
    img = T.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def images_fuse(alpha, image1, image2):
    img_fuse = (1-alpha)*image1 + alpha*image2
    return img_fuse

def consine_distance(list1,list2):
    m = len(list1)
    n = len(list1)
    consine_matrix = torch.zeros(m,n)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(m):
        for j in range(n):
            consine_matrix[i,j] = cos(list1[i],list2[j])
    return consine_matrix



# def show_image(image_tensor):
#     img_numpy = image_tensor.numpy()
#     plt.imshow(image_numpy)
#     # plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
def save_as_image(tensor, filename, nrow=8, padding=2,normalize=False, range=None, scale_each=False, pad_value=0):
    # assert (len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1)
    # input_tensor = image_tensor.clone().detach.to(torch.device('cpu'))
    #input_tensor = input_tensor.squeeze(0)
    # grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
    #                  normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor#.mul_(255)#.add_(0.5)#.clamp_(0, 255)#.permute(1, 2, 0)
    ndarr = ndarr.to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
    # torchvision.utils.save_image(image_tensor,filename,nrow,padding)

# class Cnn_net(nn.Module):
#     def __init__(self):
#         super(Cnn_net,self).__init__()
#     self.conv2 = 

#     def forward(x):
#         x = 

# image_path0 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_002_000001.jpg"
# image_path1 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_002_000002.jpg"
# image_path0 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_002_000001.jpg"
# image_path1 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_002_000002.jpg"
# image_path2 = "/home/daifengqi/gitrep/kd3a/data/images/infograph_002_000001.jpg"
# image_path3 = "/home/daifengqi/gitrep/kd3a/data/images/infograph_002_000002.jpg"
# image_path4 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_014_000001.jpg"
# image_path5 = "/home/daifengqi/gitrep/kd3a/data/images/clipart_014_000002.jpg"

image_path0 = "/home/daifengqi/gitrep/kd3a/data/images/real_002_000001.jpg"
image_path1 = "/home/daifengqi/gitrep/kd3a/data/images/real_002_000002.jpg"
image_path2 = "/home/daifengqi/gitrep/kd3a/data/images/painting_002_000001.jpg"
image_path3 = "/home/daifengqi/gitrep/kd3a/data/images/painting_002_000002.jpg"
image_path4 = "/home/daifengqi/gitrep/kd3a/data/images/real_006_000001.jpg"
image_path5 = "/home/daifengqi/gitrep/kd3a/data/images/real_006_000002.jpg"

source_image_list = [image_path0,image_path1,image_path2,image_path3,image_path4,image_path5]
source_tensor_list = []
for item in source_image_list:
    image = load_image(item,224,500)
    img_tensor = image_to_tensor(image)
    img_tensor = torch.flatten(img_tensor,start_dim=1)
    source_tensor_list.append(img_tensor)
    # print("item1",img_tensor.shape)

#the matrix[11,12,21,22]are same_class same_domain,diff_class same_domain,same_class diff_domain, diff_class diff_domain
# source_tensor_list1 = source_tensor_list[:4]
# source_tensor_list2 = source_tensor_list[4:]
# consine_matrix = consine_distance(source_tensor_list1,source_tensor_list2)
#
consine_matrix = consine_distance(source_tensor_list,source_tensor_list)
print(consine_matrix)
ipdb.set_trace()

# image_numpy_crop0 = load_image(image_path0,224,500)
# image_numpy_crop1 = load_image(image_path1,224,500)
# image_tensor0 = image_to_tensor(image_numpy_crop0)
# image_tensor1 = image_to_tensor(image_numpy_crop1)
print(image_tensor0.shape)
print(image_tensor1.shape)

image_tensor_net = images_fuse(0.3,image_tensor0,image_tensor1)
out_path = "/home/daifengqi/gitrep/kd3a/result_img/clipart_002_000001.jpg"

torchvision.utils.save_image(image_tensor_net,out_path,1,0)

