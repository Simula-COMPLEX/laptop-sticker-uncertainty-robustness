from pypylon import pylon
import numpy as np
import cv2 as cv
import json
from model_def_pl import StickerDetector
import torch
from torchvision import transforms
import onnx
import onnxruntime
import psutil

import time

NUM_CLASSES = 3  # logo + sticker + background

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
LEARNING_RATE = 0.005
BATCH_SIZE = 6

CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE
}

# MODEL_NAME = 'retinanet_resnet50_fpn'
# CHECKPOINT_PATH = 'checkpoints/retinanet_resnet50_fpn/epoch=30-step=7471.ckpt'

# MODEL_NAME = 'retinanet_resnet50_fpn_v2'
# CHECKPOINT_PATH = 'checkpoints/retinanet_resnet50_fpn_v2/epoch=31-step=7712.ckpt'
#
# MODEL_NAME = 'fasterrcnn_resnet50_fpn'
# CHECKPOINT_PATH = 'checkpoints/fasterrcnn_resnet50_fpn/epoch=14-step=1815.ckpt'

MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'
CHECKPOINT_PATH = 'checkpoints/fasterrcnn_resnet50_fpn_v2/epoch=16-step=8177.ckpt'


# MODEL_NAME = 'ssd300_vgg16'
# CHECKPOINT_PATH = 'checkpoints/ssd300_vgg16/epoch=33-step=4114.ckpt'
#
# MODEL_NAME = 'ssdlite320_mobilenet_v3_large'
# CHECKPOINT_PATH = 'checkpoints/ssdlite320_mobilenet_v3_large/epoch=18-step=9139.ckpt'


def load_camera_calibration():
    with open('calibration.json', 'r') as f:
        parameters = json.load(f)
        p = np.array(parameters['mtx'])
        d = np.array(parameters['dist'])

        # calculate undistortion map
        dist_maps = cv.initUndistortRectifyMap(p, d, np.eye(3), p, (2448, 2048), cv.CV_32FC1)

    return p, d, dist_maps


def grab_one_image(camera):
    # # grabbed = camera.GrabOne(100)
    # grabbed = camera.RetrieveResult(2000)

    # grap the latest image from the camera
    start = time.time()
    grabbed = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)

    image = grabbed.GetArray()
    print(time.time() - start)
    return image


def init_camera():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    for device in devices:
        print(device.GetFriendlyName())
    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BGR8")
    # set camera exposure
    camera.ExposureTime.SetValue(15000)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    print("Using device ", camera.GetDeviceInfo().GetModelName())
    # print camera resolution
    print("Camera resolution: ", camera.Width.GetValue(), "x", camera.Height.GetValue())

    return camera


def load_model(runtime_type, model_name=MODEL_NAME, checkout_path=CHECKPOINT_PATH):
    print("Using CUDA: ", torch.cuda.is_available())

    model = None
    if runtime_type == 'onnx':
        # for onnx model
        # export_model_path = "retinanet_resnet50_fpn_v2_aug.onnx"
        export_model_path = "./checkpoints/retinanet_resnet50_fpn_v2_aug/version_0/checkpoints" \
                            "/retinanet_resnet50_fpn_v2_aug.onnx"

        assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        # print(onnxruntime.get_available_providers())

        sess_options = onnxruntime.SessionOptions()

        # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
        # Note that this will increase session creation time so enable it for debugging only.

        # Please change the value according to best setting in Performance Test Tool result.
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        model = onnxruntime.InferenceSession(export_model_path, sess_options,
                                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    elif runtime_type == 'normal':
        # for pytorch model
        model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, model_name=model_name)
        model = model.load_from_checkpoint(checkpoint_path=checkout_path)

        # model.cuda()
        model.eval()

    return model


def run_model(image, model, runtime_type):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transform(image.float())

    if runtime_type == 'normal':
        # for pytorch model
        image = [image]
        # image[0] = image[0].cuda()
        image[0] = image[0]
        image = tuple(image)

        preds = model(image)

    elif runtime_type == 'onnx':
        # for onnx model
        image = image.numpy()
        image = image.astype(np.float32)
        # image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        start = time.time()
        preds = model.run(None, {'input': image})
        print("Time taken for inference: ", time.time() - start)
        # convert each array in pred to tensor
        preds = [torch.from_numpy(pred) for pred in preds]
        # make a dict with the tensor preds with keys boxes, scores, labels
        preds = [{'boxes': preds[0], 'scores': preds[1], 'labels': preds[2]}]

    return preds


def calc_3d_point(box, p):
    # find xy, for center of box
    x = int((box[0] + box[2]) / 2)
    y = int((box[1] + box[3]) / 2)
    # make homogenous
    center_2d_homo = np.array([x, y, 1])
    # convert to 3d. remember to invert P
    center_3d = np.matmul(np.linalg.inv(p), center_2d_homo)
    # remove z
    center_3d = center_3d[:2]
    # convert to cm
    center_3d = center_3d * 100
    center_3d_str = str(str('%.2f' % center_3d[0]) + ',' + str('%.2f' % center_3d[1]))
    return center_3d_str, center_3d.tolist()
