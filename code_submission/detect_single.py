"""
Detect a pill in a single image.
"""
import os
import sys
from pathlib import Path
from torch import no_grad, from_numpy, tensor
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class Detector:
    """
    Detector which uses pretrained model to find pills and classify them as present and missing
    """

    @no_grad()
    def __init__(self,
                 weights=ROOT / 'best.pt',  # model.pt path(s)
                 data=ROOT / 'pills.yaml',
                 img_size=(288, 288),  # inference size (height, width)
                 device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 ):

        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        self.img_size = check_img_size(img_size, s=self.model.stride)  # check image size
        self.model.warmup(imgsz=(1, 3, *self.img_size))

    @no_grad()
    def detect(self,
               image_array,
               conf_thres=0.67,  # confidence threshold
               iou_thres=0.45,
               ):
        """
        Convert image into right format and count amount of present and missing pills.
        :param image_array: the image in the form of an numpy array
        :param conf_thres: confidence threshold
        :param iou_thres: IOU threshold
        :return: a list with the amount of present pills and the amount of missing pills
                 plus their position and confidence
        """
        original_size = image_array.shape

        img = letterbox(image_array, self.img_size, stride=self.model.stride, auto=True)[0]
        img = from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img[None]
        img = img.permute(0, 3, 1, 2)

        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=25)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_size).round()

        pills = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(tensor(xyxy).view(1, 4))).view(-1).tolist()
            xywh[1] = original_size[1] - xywh[1]
            line = (cls, *xywh, conf)
            pills.append(line)

        return pills  # array with tuples (class, x, y, w, h, confidence)

    def get_model(self):
        """
        returns the model to detect with
        :return: the model
        """
        return self.model
