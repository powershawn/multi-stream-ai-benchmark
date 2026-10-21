"""Six independent, GPU-resident YOLO models, owned by one inference thread."""

from pathlib import Path
import time


class YoloModels:
    def __init__(self, config, max_batch=1):
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
        from .pipeline import VideoReader

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: check NVIDIA driver and Docker --gpus all; CPU fallback is disabled')
        if config.gpu_device >= torch.cuda.device_count():
            raise ValueError('gpu_device is outside visible CUDA devices')
        torch.cuda.set_device(config.gpu_device)
        torch.set_num_threads(config.torch_threads)
        cv2.setNumThreads(1)
        self.torch, self.config = torch, config
        self.models, self.preflight = [], []
        self.graph = None
        self.timing_scope = 'predict_wall'
        self.reset_measurements()
        self.metadata = dict(torch=torch.__version__, cuda=torch.version.cuda,
                             ultralytics=ultralytics.__version__, opencv=cv2.__version__,
                             gpu_name=torch.cuda.get_device_name(config.gpu_device),
                             gpu_total_mb=torch.cuda.get_device_properties(config.gpu_device).total_memory / 1024**2)
        reader = VideoReader(config.video, config.target_fps)
        try:
            self.metadata['video'] = dict(fps=reader.fps, frames=reader.frames,
                                          width=int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          height=int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                          duration_s=reader.frames / reader.fps)
            frame = reader.read(0)
        finally:
            reader.close()
        for index, weight in enumerate(config.weights):
            if not Path(weight).is_file():
                raise FileNotFoundError(f'model {index}: missing {weight}; run prepare first')
            print(f'Loading model {index + 1}/6: {weight}', flush=True)
            try:
                model = YOLO(weight).to(config.gpu_device)
                self.models.append(model)
                results = self._predict(model, [frame])
                torch.cuda.synchronize(config.gpu_device)
                self.preflight.append(dict(model=index, weights=weight, status='ok',
                                           detections=len(results[0].boxes) if results[0].boxes is not None else 0))
            except Exception as exc:
                raise RuntimeError(f'model {index + 1}/6 preflight failed: {exc}') from exc
        self.metadata['execution'] = config.execution
        if config.execution == 'cuda_graph':
            from .cuda_graph import CudaGraphExecutor
            if any(model.task != 'detect' for model in self.models):
                raise ValueError('cuda_graph supports YOLO detection models; use execution=predict for other tasks')
            with torch.inference_mode():
                example = self.models[0].predictor.preprocess([frame])
                self.graph = CudaGraphExecutor([model.model for model in self.models], example, max_batch)
            self.timing_scope = 'group_cuda_graph'
            self.metadata.update(graph_capture_count=self.graph.capture_count,
                                 tensor_shape=list(example.shape), independent_models=6)
            print(f'Prepared six-model CUDA graphs for batch sizes 1..{max_batch}', flush=True)

    def _predict(self, model, images):
        return model.predict(source=images, device=self.config.gpu_device,
                             imgsz=self.config.imgsz, half=self.config.half,
                             conf=.25, iou=.7, max_det=300, verbose=False,
                             save=False, stream=False)

    def infer(self, images):
        if self.graph is not None:
            return self._infer_graph(images)
        started = time.perf_counter()
        times = []
        for model in self.models:
            start = time.perf_counter()
            self._predict(model, images)
            # Include CUDA completion, preprocessing and NMS, not only enqueue time.
            self.torch.cuda.synchronize(self.config.gpu_device)
            times.append((time.perf_counter() - start) * 1000)
        self.measurements.append(dict(total_wall_ms=(time.perf_counter() - started) * 1000))
        self.evaluated_frames += len(images)
        return times

    def _infer_graph(self, images):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes
        start = time.perf_counter()
        with self.torch.inference_mode():
            # The same image preprocessing is shared; every network still executes.
            tensor = self.models[0].predictor.preprocess(images)
            prepared = time.perf_counter()
            predictions, gpu_start, gpu_end = self.graph.replay(tensor)
            submitted = time.perf_counter()
            self.last_detections = []
            for prediction in predictions:
                # Ultralytics NMS changes raw coordinates in place. Keep graph
                # buffers intact, including when validation reads them again.
                detections = non_max_suppression(prediction.clone(), conf_thres=.25,
                                                iou_thres=.7, max_det=300)
                if len(detections) != len(images):
                    raise RuntimeError('postprocessing did not return every input image')
                for detection, image in zip(detections, images):
                    scale_boxes(tensor.shape[2:], detection[:, :4], image.shape)
                self.last_detections.append(detections)
            if len(self.last_detections) != 6:
                raise RuntimeError('all six independent model results are required')
            # One explicit completion fence per batch, AFTER all six forwards,
            # all six NMS operations and original-image coordinate restoration.
            self.torch.cuda.synchronize(self.config.gpu_device)
            end = time.perf_counter()
        self.measurements.append(dict(input_preparation_wall_ms=(prepared - start) * 1000,
                                      graph_submit_wall_ms=(submitted - prepared) * 1000,
                                      group_forward_gpu_ms=gpu_start.elapsed_time(gpu_end),
                                      postprocess_and_wait_wall_ms=(end - submitted) * 1000,
                                      total_wall_ms=(end - start) * 1000))
        self.evaluated_frames += len(images)
        # The graph records the whole group. Do not invent per-model wall times.
        return [None] * 6

    def reset_measurements(self):
        self.measurements = []
        self.evaluated_frames = 0

    def performance_summary(self):
        import statistics
        from .metrics import percentile
        values = {}
        if self.measurements:
            for key in self.measurements[0]:
                samples = [row[key] for row in self.measurements]
                values[key] = dict(mean=statistics.mean(samples), p95=percentile(samples, 95))
        return dict(batch_calls=len(self.measurements), components=values,
                    evaluated_frames_per_model=[self.evaluated_frames] * 6,
                    note='CPU wall and GPU intervals overlap; component columns are not additive')
