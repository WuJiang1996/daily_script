import torch
import torchvision
import cv2
import onnx
import numpy as np
import matplotlib.pyplot as plt
import timm
import os

# print(torch.__version__)
# print(cv2.__version__)
# print(np.__version__)
# print(onnx.__version__)
classes = None
class_file = r"./classification_classes_ILSVRC2012.txt"
with open(class_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# print(classes)
def init_model(model_name):
    if model_name=='alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    if model_name=='densnet':
        model = torchvision.models.densenet121(pretrained=True)
    if model_name=='resnet':
        model = torchvision.models.resnet18(pretrained=True)
    if model_name=='mobilenet':    
        model = torchvision.models.mobilenet_v2(pretrained=True)
    if model_name=='squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
    if model_name=='inception':
        model = torchvision.models.inception_v3(pretrained=False)
    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained=True)
    if model_name=='vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    if model_name=='vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    if model_name=='shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    if model_name=='cspdarknet53':
        model = timm.create_model('cspdarknet53', pretrained=True)
    if model_name=='seresnet18':
        model = timm.create_model('seresnet18',pretrained=True)
    if model_name=='senet154':
        model = timm.create_model('senet154', pretrained=True)
    if model_name=='seresnet50':
        model = timm.create_model('seresnet50',pretrained=True)
    if model_name=='resnest50d':
        model = timm.create_model('resnest50d', pretrained=True)
    if model_name=='skresnet50':
        model = timm.create_model('skresnet50',pretrained=True)
    model.eval()
    if model_name=='inception':
        dummy = torch.randn(1,3,299,299)
    else:
        dummy = torch.randn(1,3,224,224)
    return model, dummy


model, dummy = init_model('resnet')

onnx_name = 'exported.onnx'
torch.onnx.export(model, dummy, onnx_name)

# 载入onnx模块
model_ = onnx.load(onnx_name)
#检查IR是否良好
onnx.checker.check_model(model_)

# opencv dnn加载
net = cv2.dnn.readNetFromONNX(onnx_name)

img_file = r"./dog.jpg"
assert os.path.exists(img_file)


#使用torchvision来推理
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image
img = Image.open(img_file)
img_t = transform(img) # 输入给模型的图像数据要先进行转换
batch_t = torch.unsqueeze(img_t, 0)
tc_out = model(batch_t).detach().numpy()
# Get a class with a highest score.
tc_out = tc_out.flatten()
classId = np.argmax(tc_out)
confidence = tc_out[classId]

label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
print(label)

#opencv调用onnx模型
frame = cv2.imread(img_file)
    
# Create a 4D blob from a frame.
inpWidth = dummy.shape[-2]
inpHeight = dummy.shape[-2]
# blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), crop=False)
#以上模型中shufflenet, resnest, skresnet模型还不能被opencv dnn模块成功加载，
# 有些op还不支持，会报错。另外需要注意的是blobFromImage函数的参数：
#对图像数据的变换要和pytorch模型设置相同，比如归一化操作，减去均值等。
blob = cv2.dnn.blobFromImage(frame,
                                scalefactor=1.0 / 255,
                                size=(inpWidth, inpHeight),
                                mean=[0.485, 0.456, 0.406],
                                swapRB=True,
                                crop=False)

# Run a model
net.setInput(blob)
out = net.forward()
print(out.shape)

# Get a class with a highest score.
out = out.flatten()
classId = np.argmax(out)
confidence = out[classId]
print(classId)

# Put efficiency information.
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
print(label)
cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

# Print predicted class.
label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
print(label)
cv2.putText(frame, label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
winName = 'onnx'

cv2.imshow(winName, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()