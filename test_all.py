import cv2
import numpy as np
import torch
from opt import parse_opt

from models.bilinear_dense import BilinearDense201
from models.bilinear_resnet import BilinearResNet101


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_backward_hook(hook_fn)

    def generate(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax()
        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.target_layer.forward(input_image).detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        grad_cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            grad_cam += w * activations[i, :, :]

        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, input_image.shape[2:])
        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)
        return grad_cam


# 用于可视化 Grad-CAM 的辅助函数
def visualize_cam(mask, image):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    opt = parse_opt()

    BD201_cam = BilinearDense201(opt.num_classes, opt.attention, None).to(device)
    BR101_cam = BilinearResNet101(opt.num_classes, opt.attention, None).to(device)

    # BD201_path = "autodl-tmp/runs_twice/save_model/densenet201_all/BAM/True/loss_stage1/best.pt"
    # BR101_path = "autodl-tmp/runs_twice/save_model/resnet101_all/BAM/True/loss_stage1/best.pt"

    # BD201_cam.load_state_dict(torch.load(BD201_path))
    # BR101_cam.load_state_dict(torch.load(BR101_path))
