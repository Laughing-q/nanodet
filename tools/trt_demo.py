from collections import OrderedDict, namedtuple
from yolo.utils.torch_utils import select_device
from nanodet.model.module.nms import multiclass_nms
from nanodet.data.transform import Pipeline
from nanodet.util import cfg, load_config, overlay_bbox_cv
from nanodet.data.transform.warp import warp_boxes
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img

# from nanodet.util.visualization import _COLORS
import torch
import os
import numpy as np
import cv2
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import time
import torch.nn.functional as F


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Predictor(object):
    def __init__(self, cfg, engine_path, device="cuda:0"):
        self.cfg = cfg
        self.device = select_device(device)
        self.bindings = OrderedDict()

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
                self.device
            )
            self.bindings[name] = Binding(
                name, dtype, shape, data, int(data.data_ptr())
            )
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        self.batch_size = self.bindings["images"].shape[0]
        self.num_classes = self.bindings["output"].shape[-1] - 4

        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        # preprocess
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        # TODO
        meta["img"] = meta["img"].transpose(2, 0, 1)
        meta["img"] = torch.from_numpy(np.ascontiguousarray(meta["img"])).to(
            self.device
        )
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32).half()
        # meta["img"] = F.pad(meta["img"], [0, 0, 0, 160], value=0)

        assert meta["img"].shape == self.bindings["images"].shape, (
            meta["img"].shape,
            self.bindings["images"].shape,
        )
        # inference
        t1 = time_sync()
        self.binding_addrs["images"] = int(meta["img"].data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        preds = self.bindings["output"].data
        t2 = time_sync()
        print(t2 - t1)
        return self.postprocess(preds, meta)

    def postprocess(self, preds, meta):
        h, w = meta["img"].shape[2:]
        scores = preds[..., 4:]  # TODO
        bboxes = preds[..., :4]
        bboxes[..., 0] = bboxes[..., 0].clamp(min=0, max=w)
        bboxes[..., 1] = bboxes[..., 1].clamp(min=0, max=h)
        bboxes[..., 2] = bboxes[..., 2].clamp(min=0, max=w)
        bboxes[..., 3] = bboxes[..., 3].clamp(min=0, max=h)
        # NMS
        result_list = []
        for i in range(self.batch_size):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)

        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result

        return meta, det_results

    def visualize(self, img, dets, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


if __name__ == "__main__":
    w = "/home/laughing/nanodet/nanodet.engine"

    load_config(cfg, "/home/laughing/nanodet/config/nanodet-plus-m_416.yml")
    img = cv2.imread("/e/datasets/贵阳银行/play_phone/guiyang0721/images/train/3_0.jpg")
    predictor = Predictor(cfg, engine_path=w, device="0")

    rtsp = "rtsp://admin:shtf123456@192.168.1.233:554/h264/ch1/main/av_stream"
    cap = cv2.VideoCapture(rtsp)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        meta, res = predictor.inference(frame)
        result_image = predictor.visualize(frame, res[0], cfg.class_names, 0.4)
        cv2.imshow("p", result_image)
        if cv2.waitKey(1) == ord("q"):
            exit()
