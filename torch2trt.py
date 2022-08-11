
# import torch
# from torch.autograd import Variable
# import onnx


# input_name = ['input']
# output_name = ['output']
# input = Variable(torch.randn(1, 3, 544, 544)).cuda()
# model = x.model.cuda()#x.model为我生成的模型

# # model = torch.load('', map_location="cuda:0")
# torch.onnx.export(model, input, 'model.onnx', input_names=input_name, output_names=output_name, verbose=True)

# #RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

# model = onnx.load("model.onnx")
# onnx.checker.check_model(model)
# print("==> Passed")


# import pycuda.autoinit
# import numpy as np
# import pycuda.driver as cuda
# import tensorrt as trt
# import torch
# import os
# import time
# from PIL import Image
# import cv2
# import torchvision

# filename = '000000.jpg'
# max_batch_size = 1
# onnx_model_path = 'yolo.onnx'

# TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


# def get_img_np_nchw(filename):
#     image = cv2.imread(filename)
#     image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_cv = cv2.resize(image_cv, (1920, 1080))
#     miu = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_np = np.array(image_cv, dtype=float) / 255.
#     r = (img_np[:, :, 0] - miu[0]) / std[0]
#     g = (img_np[:, :, 1] - miu[1]) / std[1]
#     b = (img_np[:, :, 2] - miu[2]) / std[2]
#     img_np_t = np.array([r, g, b])
#     img_np_nchw = np.expand_dims(img_np_t, axis=0)
#     return img_np_nchw

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         """Within this context, host_mom means the cpu memory and device means the GPU memory
#         """
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()


# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()
#     for binding in engine:
#         size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         # Allocate host and device buffers
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#         # Append the device buffer to device bindings.
#         bindings.append(int(device_mem))
#         # Append to the appropriate list.
#         if engine.binding_is_input(binding):
#             inputs.append(HostDeviceMem(host_mem, device_mem))
#         else:
#             outputs.append(HostDeviceMem(host_mem, device_mem))
#     return inputs, outputs, bindings, stream


# def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
#                fp16_mode=False, int8_mode=False, save_engine=False,
#                ):
#     """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

#     def build_engine(max_batch_size, save_engine):
#         """Takes an ONNX file and creates a TensorRT engine to run inference with"""
#         EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#         with trt.Builder(TRT_LOGGER) as builder, \
#                 builder.create_network(EXPLICIT_BATCH) as network, \
#                 trt.OnnxParser(network, TRT_LOGGER) as parser:

#             builder.max_workspace_size = 1 << 30  # Your workspace size
#             builder.max_batch_size = max_batch_size
#             # pdb.set_trace()
#             builder.fp16_mode = fp16_mode  # Default: False
#             builder.int8_mode = int8_mode  # Default: False
#             if int8_mode:
#                 # To be updated
#                 raise NotImplementedError

#             # Parse model file
#             if not os.path.exists(onnx_file_path):
#                 quit('ONNX file {} not found'.format(onnx_file_path))

#             print('Loading ONNX file from path {}...'.format(onnx_file_path))
#             with open(onnx_file_path, 'rb') as model:
#                 print('Beginning ONNX file parsing')
#                 parser.parse(model.read())

#                 if not parser.parse(model.read()):
#                     for error in range(parser.num_errors):
#                         print(parser.get_error(error))
#                     print("===========Parsing fail!!!!=================")
#                 else :
#                     print('Completed parsing of ONNX file')

#             print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

#             engine = builder.build_cuda_engine(network)
#             print("Completed creating Engine")

#             if save_engine:
#                 with open(engine_file_path, "wb") as f:
#                     f.write(engine.serialize())
#             return engine

#     if os.path.exists(engine_file_path):
#         # If a serialized engine exists, load it instead of building a new one.
#         print("Reading engine from file {}".format(engine_file_path))
#         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#     else:
#         return build_engine(max_batch_size, save_engine)


# def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
#     # Transfer data from CPU to the GPU.
#     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#     # Run inference.
#     context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#     # Synchronize the stream
#     stream.synchronize()
#     # Return only the host outputs.
#     return [out.host for out in outputs]


# def postprocess_the_outputs(h_outputs, shape_of_output):
#     h_outputs = h_outputs.reshape(*shape_of_output)
#     return h_outputs



# img_np_nchw = get_img_np_nchw(filename)
# img_np_nchw = img_np_nchw.astype(dtype=np.float32)

# # These two modes are dependent on hardwares
# fp16_mode = False
# int8_mode = False
# trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
# # Build an engine
# engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
# # Create the context for this engine
# context = engine.create_execution_context()
# # Allocate buffers for input and output
# inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# # Do inference
# shape_of_output = (max_batch_size, 1000)
# # Load data to the buffer
# inputs[0].host = img_np_nchw.reshape(-1)

# # inputs[1].host = ... for multiple input
# t1 = time.time()
# trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
# t2 = time.time()
# feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

# print('TensorRT ok')
# #将model改为自己的模型,此处为pytoch的resnet50,需联网下载
# model = torchvision.models.resnet50(pretrained=True).cuda()
# resnet_model = model.eval()

# input_for_torch = torch.from_numpy(img_np_nchw).cuda()
# t3 = time.time()
# feat_2= resnet_model(input_for_torch)
# t4 = time.time()
# feat_2 = feat_2.cpu().data.numpy()
# print('Pytorch ok!')


# mse = np.mean((feat - feat_2)**2)
# print("Inference time with the TensorRT engine: {}".format(t2-t1))
# print("Inference time with the PyTorch model: {}".format(t4-t3))
# print('MSE Error = {}'.format(mse))

# print('All completed!')



#使用 trt的Python API 构建 一个对输入做池化的简单网络（直接通过 TensorRT 的 API 逐层搭建网络并序列化模）
import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    pool = network.add_pooling(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    pool.stride = (2, 2)
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]]*3) 
    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")



#使用 Python API 转换（中间表示的模型（如 ONNX）转换成 TensorRT）
import torch
import onnx
import tensorrt as trt

onnx_model = 'model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        return self.pool(x)

device = torch.device('cuda:0')

# generate ONNX model
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11)
onnx_model = onnx.load(onnx_model)

# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
config.max_workspace_size = 1<<20
profile = builder.create_optimization_profile()

profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])
config.add_optimization_profile(profile)
# create engine
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

with open('model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")




#使用 C++ API 转换

#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <../samples/common/logger.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

int main(int argc, char** argv)
{
        // Create builder
        Logger m_logger;
        IBuilder* builder = createInferBuilder(m_logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        // Parse ONNX file
        IParser* parser = nvonnxparser::createParser(*network, m_logger);
        bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

        // Get the name of network input
        Dims dim = network->getInput(0)->getDimensions();
        if (dim.d[0] == -1)  // -1 means it is a dynamic model
        {
                const char* name = network->getInput(0)->getName();
                IOptimizationProfile* profile = builder->createOptimizationProfile();
                profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                config->addOptimizationProfile(profile);
        }


        // Build engine
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Serialize the model to engine file
        IHostMemory* modelStream{ nullptr };
        assert(engine != nullptr);
        modelStream = engine->serialize();

        std::ofstream p("model.engine", std::ios::binary);
        if (!p) {
                std::cerr << "could not open output file to save model" << std::endl;
                return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "generate file success!" << std::endl;

        // Release resources
        modelStream->destroy();
        network->destroy();
        engine->destroy();
        builder->destroy();
        config->destroy();
        return 0;
}


#使用 Python API 推理
from typing import Union, Optional, Sequence,Dict,Any

import torch
import tensorrt as trt

class TRTWrapper(torch.nn.Module):
    def __init__(self,engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs

model = TRTWrapper('model.engine', ['output'])
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda()))
print(output)



#使用 C++ API 推理


#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <../samples/common/logger.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);


void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
        const ICudaEngine& engine = context.getEngine();

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(IN_NAME);
        const int outputIndex = engine.getBindingIndex(OUT_NAME);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{ nullptr };
        size_t size{ 0 };

        std::ifstream file("model.engine", std::ios::binary);
        if (file.good()) {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, size);
                file.close();
        }

        Logger m_logger;
        IRuntime* runtime = createInferRuntime(m_logger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        // generate input data
        float data[BATCH_SIZE * 3 * IN_H * IN_W];
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++)
                data[i] = 1;

        // Run inference
        float prob[BATCH_SIZE * 3 * IN_H * IN_W /4];
        doInference(*context, data, prob, BATCH_SIZE);

        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return 0;
}
