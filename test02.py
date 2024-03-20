import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）

model_path = "/data2/wj/ChatGLM-6B-main/ptuning/THUDM/chatglm-6b/dataroot/models/THUDM/chatglm-6b"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 微调前
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# model = model.eval()

# response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
# print("response_before: ", response)

# Fine-tuning 后的表现测试
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
# 此处使用你的 ptuning 工作目录
prefix_state_dict = torch.load(os.path.join("/data2/wj/ChatGLM-6B-main/ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000", "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

#V100 机型上可以不进行量化
#print(f"Quantized to 4 bit")
#model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
print("response_later: ", response)
