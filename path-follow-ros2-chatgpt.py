import os, ctypes, cv2, rclpy
from rclpy.node import Node

# choose your camera source:
USE_CSI = False  # set True if using CSI camera on Orin
CAM_INDEX = 0    # for USB cams

def gstreamer_csi_pipeline(width=1280, height=720, fps=30):
    # For CSI cameras on Orin/Jetson
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

class PublishThread:
    # stub – keep your existing implementation
    def __init__(self, node, repeat):
        self.node = node
        self._running = True
    def update(self, x,y,z,th,speed,turn): pass
    def stop(self): self._running = False
    def wait_for_subscribers(self): pass

class YoLov5TRT:
    # Minimal skeleton: replace with your real wrapper
    def __init__(self, engine_path):
        import tensorrt as trt
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.input_name = "input"   # adapt to your network
        self.output_name = "output" # adapt to your network

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Load plugins BEFORE runtime creation
        plugin_path = os.path.join("build", "libmyplugins.so")
        if os.path.exists(plugin_path):
            try:
                ctypes.CDLL(plugin_path)
            except Exception as e:
                print(f"[WARN] Failed to load plugin library: {e}")

        logger = trt.Logger(trt.Logger.INFO)
        with trt.Runtime(logger) as runtime, open(engine_path, "rb") as f:
            engine_bytes = f.read()
            # deserialize returns None on magicTag/version mismatch
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine. Rebuild engine & plugins on this device/version.")

        # Create context
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context() failed.")

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.cuda = cuda

        # Prepare I/O buffers (explicit batch; single static shape)
        self.stream = cuda.Stream()

        idx_in = self.engine.get_binding_index(self.input_name)
        idx_out = self.engine.get_binding_index(self.output_name)
        if idx_in < 0 or idx_out < 0:
            # Try TensorRT 8.6 names if you used network-defined tensor names
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            raise RuntimeError(f"Binding names not found. Available: {names}")

        self.input_idx = idx_in
        self.output_idx = idx_out

        in_shape = self.engine.get_binding_shape(idx_in)  # e.g. (1,3,640,640)
        out_shape = self.engine.get_binding_shape(idx_out)

        # If dynamic, set input shape
        if -1 in in_shape:
            # Set your real shape here
            in_shape = (1, 3, 640, 640)
            self.context.set_binding_shape(idx_in, in_shape)

        self.in_size = abs(int(self._vol(in_shape)))
        self.out_size = abs(int(self._vol(out_shape)))

        self.d_input = cuda.mem_alloc(self.in_size * 4)
        self.d_output = cuda.mem_alloc(self.out_size * 4)
        self.bindings = [None] * self.engine.num_bindings
        self.bindings[idx_in] = int(self.d_input)
        self.bindings[idx_out] = int(self.d_output)

    def _vol(self, shape):
        v = 1
        for s in shape:
            v *= s
        return v

    def infer(self, bgr_image):
        import numpy as np
        # Preprocess to NCHW float32 [0,1]
        img = cv2.resize(bgr_image, (640, 640))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype('float32') / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, 0).copy() # NCHW contiguous

        # H2D
        self.cuda.memcpy_htod_async(self.d_input, x, self.stream)
        # Inference
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # D2H
        import numpy as np
        out = np.empty(self.out_size, dtype=np.float32)
        self.cuda.memcpy_dtoh_async(out, self.d_output, self.stream)
        self.stream.synchronize()

        # TODO: decode out -> draw boxes on bgr_image
        return bgr_image  # placeholder: draw your post-proc results here

    def destroy(self):
        try:
            if self.d_input: self.d_input.free()
        except: pass
        try:
            if self.d_output: self.d_output.free()
        except: pass
        self.context = None
        self.engine = None
        self.stream = None

class YOLOv5Node(Node):
    def __init__(self):
        super().__init__('yolov5_node')
        self.pub_thread = None
        self.yolov5_wrapper = None
        self.cap = None
        self.vid_writer = None
        self.timer = None

        try:
            # Parameters
            self.declare_parameter('speed', 0.5)
            self.declare_parameter('turn', 1.0)
            self.declare_parameter('repeat_rate', 0.0)
            self.declare_parameter('engine_path', 'build/yolov5s.engine')
            self.declare_parameter('save_path', 'save/video_out.avi')

            repeat = float(self.get_parameter('repeat_rate').value)
            self.pub_thread = PublishThread(self, repeat)

            engine_file_path = str(self.get_parameter('engine_path').value)
            if not os.path.exists(engine_file_path):
                raise FileNotFoundError(f"Engine file not found: {engine_file_path}")

            # Create TRT wrapper (loads plugins internally)
            self.yolov5_wrapper = YoLov5TRT(engine_file_path)

            # Camera
            if USE_CSI:
                pipeline = gstreamer_csi_pipeline()
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(CAM_INDEX)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0: fps = 30.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            save_path = str(self.get_parameter('save_path').value)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

            # Timer (10 Hz)
            self.timer = self.create_timer(0.1, self.process_frame)

            self.get_logger().info("YOLOv5 node initialized successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize node: {e}")
            self.cleanup()
            raise

    def process_frame(self):
        ret, image = self.cap.read() if self.cap else (False, None)
        if not ret or image is None:
            self.get_logger().error("Failed to capture image")
            return

        try:
            img_out = self.yolov5_wrapper.infer(image)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return

        # Show & save
        cv2.imshow("result", img_out)
        self.vid_writer.write(img_out)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            self.get_logger().info("Quit requested.")
            rclpy.shutdown()

        # Publish control (replace x,y,z,th with your actual logic)
        x=y=z=th=0.0
        self.pub_thread.update(
            x,y,z,th,
            float(self.get_parameter('speed').value),
            float(self.get_parameter('turn').value)
        )

    def cleanup(self):
        try:
            if self.timer is not None:
                self.timer.cancel()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.vid_writer is not None:
                self.vid_writer.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if self.yolov5_wrapper is not None:
                self.yolov5_wrapper.destroy()
        except Exception:
            pass
        try:
            if self.pub_thread is not None:
                self.pub_thread.stop()
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        print("create YOLOv5Node++++++")
        node = YOLOv5Node()
        try:
            node.pub_thread.wait_for_subscribers()
        except Exception as e:
            node.get_logger().warn(f"Waiting for subscribers failed: {e}")

        node.get_logger().info("YOLOv5 TensorRT node started")
        node.get_logger().info(f"speed={node.get_parameter('speed').value}, turn={node.get_parameter('turn').value}")
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node: node.get_logger().info("Node stopped by user")
        else: print("Node stopped by user")
    except Exception as e:
        if node: node.get_logger().error(f"Error: {e}")
        else: print(f"Error during node initialization: {e}")
    finally:
        try:
            if node is not None:
                node.cleanup()
                node.destroy_node()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        rclpy.shutdown()

if __name__ == "__main__":
    main()
