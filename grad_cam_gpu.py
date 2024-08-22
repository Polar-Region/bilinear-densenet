import os
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from utils import visualize_cam, Normalize
from GradCam import GradCAM, GradCAMpp

resnet = models.resnet101(pretrained=True)
resnet.eval(), resnet.cuda()

densenet = models.densenet161(pretrained=True)
densenet.eval(), densenet.cuda()

cam_dict = dict()


resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))
densenet_gradcam = GradCAM(densenet_model_dict, True)
densenet_gradcampp = GradCAMpp(densenet_model_dict, True)
cam_dict['densenet'] = [densenet_gradcam, densenet_gradcampp]

img_dir = 'autodl-tmp/data/train'  # 图片文件夹
output_dir = 'outputs'  # 输出文件夹
os.makedirs(output_dir, exist_ok=True)  # 确保输出文件夹存在

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_imgs = []

# 列出图片文件夹中的所有子文件夹
subfolders = [os.path.join(img_dir, d) for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

# 遍历每个子文件夹
for folder in subfolders:
    # 创建对应的输出子文件夹
    output_subfolder = os.path.join(output_dir, os.path.basename(folder))
    os.makedirs(output_subfolder, exist_ok=True)

    # 列出当前子文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(image_files) > 0:
        # 如果有图片文件，则加载第一张图片
        img_path = os.path.join(folder, image_files[0])
        pil_img = PIL.Image.open(img_path)

        # 将PIL图像转换为PyTorch张量并进行归一化和调整大小
        torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
        torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
        normed_torch_img = normalizer(torch_img)
        torch_imgs.append(normed_torch_img)

        # 构建图像集合
        images = []
        for gradcam, gradcam_pp in cam_dict.values():
            mask, _ = gradcam(normed_torch_img.cuda())
            heatmap, result = visualize_cam(mask.cpu(), torch_img)

            mask_pp, _ = gradcam_pp(normed_torch_img.cuda())
            heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img)

            images.append(torch.stack([torch_img.squeeze().cpu(), result, result_pp], 0))

        # 创建图像网格
        images = make_grid(torch.cat(images, 0), nrow=3)

        # 保存图像到对应的子文件夹中
        output_name = 'output_image.jpg'  # 输出图像的文件名
        output_path = os.path.join(output_subfolder, output_name)
        save_image(images, output_path)

        # 打开并显示输出图像
        output_pil_img = PIL.Image.open(output_path)
        output_pil_img.show()
