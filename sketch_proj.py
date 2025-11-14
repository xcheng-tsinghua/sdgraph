from flask import Flask, request, jsonify
from encoders_3rd import sketch_rnn
import torch
import os
app = Flask(__name__)


# 加载模型
sketch_rnn_model = sketch_rnn.SketchRNN().cuda()
model_save = './model_trained/sketchrnn_proj.pth'
model_save = os.path.abspath(model_save)
sketch_rnn_model.load_state_dict(torch.load(model_save))
print('load weight from ', model_save)

# 设置为评估模式
sketch_rnn_model = sketch_rnn_model.eval()

# 构建采样器
sampler = sketch_rnn.Sampler(sketch_rnn_model)


@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    x = torch.tensor(data['input'], dtype=torch.float32)
    print('receive data: ', x)
    y = sampler.sample_s3(x).tolist()
    print('inference data: ', y)

    return jsonify({"output": y})


def test_input():
    input_len = 20
    exam_input = torch.rand(input_len, 2)
    exam_input_flag = torch.randint(0, 2, (input_len, 1))
    exam_input = torch.cat([exam_input, exam_input_flag], dim=1)
    x = torch.tensor(exam_input, dtype=torch.float32)
    y = sampler.sample_s3(x)

    print(y.size())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # test_input()




