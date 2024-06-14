import torch
import numpy as np
import Indrnn_action_network
from torch.autograd import Variable

# 定义标签和汉字对应关系
label_to_chinese = {
    0: '抓',
    1: '点击',
    2: '扩大',
    3: '捏',
    4: '顺时针旋转',
    5: '逆时针旋转',
    6: '向右滑动',
    7: '向左滑动',
    8: '向上滑动',
    9: '向下滑动',
    10: '滑动X',
    11: '滑动V',
    12: '滑动+',
    13: '摇'
}

indim = 22
outputclass = 14
model = Indrnn_action_network.stackedIndRNN_encoder(indim, outputclass)
ckpt = torch.load("indrnn_action_model_state.pkl", map_location='cpu')
model.load_state_dict(ckpt, strict=False)

inputs = np.load('nie_output/all_landmark_data.npy')
#nputs = inputs.reshape(-1,12,22,6)

inputs = inputs.transpose(1, 0, 2, 3)
inputs = Variable(torch.from_numpy(inputs).float())
out = model(inputs)

print(label_to_chinese[out.detach().numpy().argmax()])