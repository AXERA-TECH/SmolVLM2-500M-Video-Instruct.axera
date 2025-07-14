# SmolVLM2-500M-Video-Instruct.axera

> HuggingFaceTB SmolVLM2-500M-Video-Instruct DEMO on Axera.

- 目前支持 `Python` 语言, `C++` 代码在开发中.
- 预编译模型可以从[百度网盘](https://pan.baidu.com/s/1udw7_IMQehr_2CmipfLOXw?pwd=n6qe)下载.
- 如需自行导出编译 `VIT` 模型请参考 [模型转换](/model_convert/README.md).

## 支持平台

- [x] AX650N
- [ ] AX630C

## Git Clone

首先使用如下命令 `clone` 本项目, 然后进入 `python` 文件夹:

```bash
$ git clone git@github.com:AXERA-TECH/SmolVLM2-500M-Video-Instruct.axera.git
$ cd SmolVLM2-500M-Video-Instruct.axera/python
```

之后在开发板上下载或安装以下支持库:

- 从 `huggingface` 下载 `SmolVLM2-500M-Video-Instruct` 模型.

    ```bash
    $ git clone https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    ```

- 在开发板上安装配置 `pyaxengine`, [点击跳转下载链接](https://github.com/AXERA-TECH/pyaxengine/releases). 注意板端 `SDK` 最低版本要求:

    - AX650 SDK >= 2.18
    - AX620E SDK >= 3.12
    - 执行 `pip3 install axengine-x.x.x-py3-none-any.whl` 安装

将下载后的预编译模型解压到当前文件夹[🔔可选], 默认文件夹排布如下:

```bash
(lerobot) ➜  python git:(master) ✗ tree -L 2 .
.
├── infer_axmodel.py
├── infer.py
├── SmolVLM2-500M-Video-Instruct
│   ├── added_tokens.json
│   ├── chat_template.json
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── onnx
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── SmolVLM2-500M-Video-Instruct_axmodel
│   ├── llama_p128_l0_together.axmodel
│   ├── llama_p128_l10_together.axmodel
│   ├── llama_p128_l11_together.axmodel
│   ├── llama_p128_l12_together.axmodel
│   ├── llama_p128_l13_together.axmodel
│   ├── llama_p128_l14_together.axmodel
│   ├── llama_p128_l15_together.axmodel
│   ├── llama_p128_l16_together.axmodel
│   ├── llama_p128_l17_together.axmodel
│   ├── llama_p128_l18_together.axmodel
│   ├── llama_p128_l19_together.axmodel
│   ├── llama_p128_l1_together.axmodel
│   ├── llama_p128_l20_together.axmodel
│   ├── llama_p128_l21_together.axmodel
│   ├── llama_p128_l22_together.axmodel
│   ├── llama_p128_l23_together.axmodel
│   ├── llama_p128_l24_together.axmodel
│   ├── llama_p128_l25_together.axmodel
│   ├── llama_p128_l26_together.axmodel
│   ├── llama_p128_l27_together.axmodel
│   ├── llama_p128_l28_together.axmodel
│   ├── llama_p128_l29_together.axmodel
│   ├── llama_p128_l2_together.axmodel
│   ├── llama_p128_l30_together.axmodel
│   ├── llama_p128_l31_together.axmodel
│   ├── llama_p128_l3_together.axmodel
│   ├── llama_p128_l4_together.axmodel
│   ├── llama_p128_l5_together.axmodel
│   ├── llama_p128_l6_together.axmodel
│   ├── llama_p128_l7_together.axmodel
│   ├── llama_p128_l8_together.axmodel
│   ├── llama_p128_l9_together.axmodel
│   ├── llama_post.axmodel
│   └── model.embed_tokens.weight.npy
├── SmolVLMVisionEmbeddings.pkl
├── utils
│   └── infer_func.py
└── vit-models
    ├── vision_model.axmodel
    └── vision_model.onnx [可选]
```

## 上板部署

- `AX650N` 的设备已预装 `Ubuntu 22.04`
- 以 `root` 权限登陆 `AX650N` 的板卡设备
- 接入互联网, 确保 `AX650N` 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备: `AX650N DEMO Board`、`爱芯派Pro(AX650N)`

### Python API 运行

#### Requirements

```bash
$ mkdir /opt/site-packages
$ cd python
$ pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后, 重新连接终端或者执行 `source ~/.bashrc`

```bash
$ export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
$ export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

在 `Axera 开发板` 上运行以下命令开启图像理解功能:

```sh
$ cd SmolVLM2-500M-Video-Instruct.axera/python
$ python3 infer_axmodel.py
```

输入图像为:

![image.png](assets/girl.png)

通过命令行参数可以手动指定图像路径, 模型推理结果如下:

```bash
$ python3 infer_axmodel.py -i ../assets/girl.png --vit_model vit-models/vision_model.axmodel

Model loaded successfully!
slice_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]
Slice prefill done: 0
Slice prefill done: 1
Slice prefill done: 2
Slice prefill done: 3
Slice prefill done: 4
Slice prefill done: 5
Slice prefill done: 6
Slice prefill done: 7
Slice prefill done: 8
answer >>  The image depicts a young woman with long, light gray hair, adorned with two pink flowers in her hair. She is standing on a beach, facing the camera with a neutral expression. The woman is wearing a blue off-shoulder dress that is open at the front, revealing^@ a white lace top underneath. She is also wearing a silver choker necklace and a silver bracelet on her left wrist.

The background of the image reveals a clear blue sky with fluffy white clouds, suggesting a sunny day. The ocean is visible in the distance, with gentle waves crashing onto the shore. The overall scene suggests a serene and peaceful beach setting.

The woman's attire and accessories, along with the serene ocean and clear sky, create a calm and picturesque atmosphere. The image does not contain any^@ discernible text or additional objects. The relative positions of the objects suggest that the woman is standing in the foreground, with the ocean and sky in the background. The image does not provide any information that would allow for a specific question to be answered definitively.
```

#### 图像理解任务·推理耗时统计

该模型一共有 `32` 层 `decode layer`, 详细耗时信息如下:

Model | Time |
---| ---|
ImageEncoder | 1830 ms |
Prefill TTFT | 2892.151 ms |
Decoder | 27.51 ms |

其中:

- `Prefill` 阶段, 每一层的 `llama_layer` 最大耗时 `90.3 ms`.

    各个子图耗时:

    ```sh
    g1: 3.143 ms
    g2: 4.909 ms
    g3: 6.610 ms
    g4: 8.263 ms
    g5: 9.997 ms
    g6: 11.819 ms
    g7: 13.579 ms
    g8: 15.096 ms
    g9: 16.814 ms
    ```

- `Decoder` 阶段, 每一层的 `llama_layer` 平均耗时 `0.780 ms` ms.
- `llama_post` 耗时 `2.551 ms`.

模型解码速度为: 1000 / 27.51 ms = 36.35 tokens/s.

## 技术讨论

- Github issues
- QQ 群: 139953715
