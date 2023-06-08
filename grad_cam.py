import os

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

frame_index = 350  # Change this


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ Example usage of using cam-methods on a VIT network.
    """

    #model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    #model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    model.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    method = "gradcam"  # Choose the desired method

    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    if method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    if method == "ablationcam":
        cam = methods[method](model=model,
                              target_layers=target_layers,
                              use_cuda=use_cuda,
                              reshape_transform=reshape_transform,
                              ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=model,
                              target_layers=target_layers,
                              use_cuda=use_cuda,
                              reshape_transform=reshape_transform)

    image_path = "./data/orig_data/train/001.mp4"  # Specify the path to the video
    vidcap = cv2.VideoCapture(image_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, image = vidcap.read()

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=False,  # Set the desired eigen_smooth value
                        aug_smooth=False)  # Set the desired aug_smooth value

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    top_class_idx = np.argmax(grayscale_cam)
    print(top_class_idx)

    output_dir = "./data/grad_images"  # Specify the output directory
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    output_path = os.path.join(output_dir, f'{method}_cam_{frame_index}.jpg')
    cv2.imwrite(output_path, cam_image)
