# 模型转换

## 环境配置

创建虚拟环境

```bash
$ conda create -n SmolVLM2-500M-Video-Instruct python=3.11 -y
$ conda activate SmolVLM2-500M-Video-Instruct
```

## 导出 Vit-ONNX 模型 (PyTorch -> ONNX)

示例命令如下:

```bash
$ python3 export_onnx.py -m /path/your/hugging_face/models/SmolVLM2-500M-Video-Instruct/ -o ./vit-models
```

其中 `-m` 参数需要指定 `hugging_face SmolVLM2-500M-Video-Instruct` 模型路径, 如果模型不存在, 可以通过以下命令下载:

```bash
$ git clone https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

模型成功导出成功后会在 `vit-models` 目录中生成所需要的 `onnx` 模型.

## 模型编译 (ONNX -> AXmodel)

使用模型转换工具 `Pulsar2` 将 `ONNX` 模型转换成适用于 `Axera-NPU` 运行的模型文件格式 `.axmodel`, 通常情况下需要经过以下两个步骤:

- 生成适用于该模型的 `PTQ` 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译）, 更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集

```sh
$ bash download_dataset.sh
```

执行结束后可以在当前文件夹内看到名为 `hidden_states.tar` 的量化文件.

### 修改配置文件
 
在 `pulsar2_configs` 目录中, 检查 `*.json` 中 `calibration_dataset` 字段, 将该字段配置的路径改为上一步下载的量化数据集存放路径, 通常可以是 `.tar` 或 `.zip` 文件.

### Pulsar2 build 编译

示例命令如下:

```bash
$ pulsar2 build --output_dir compiled_output --config pulsar2_configs/config.json  --npu_mode NPU3 --input vit-models/vision_model.onnx  --compiler.check 0
```

关于 `pulsar2 build` 更详细的文档请参考 [Pulsar2-QuickStart](https://npu.pages-git-ext.axera-tech.com/pulsar2-docs/user_guides_quick/quick_start_ax650.html).

注意, 由于 `calibration` 数据集数量较少, 因此精度可能存在问题.

### 大模型编译

```
pulsar2 llm_build --input_path /data/tmp/yongqiang/nfs/SmolVLM2-500M-Video-Instruct.axera/python/SmolVLM2-500M-Video-Instruct --output_path /data/tmp/yongqiang/nfs/SmolVLM2-500M-Video-Instruct.axera/python/SmolVLM2-500M-Video-Instruct_axmodel --hidden_state_type bf16 --prefill_len 128 --kv_cache_len 2559 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 --last_kv_cache_len 640 --last_kv_cache_len 768 --last_kv_cache_len 896 --last_kv_cache_len 1024  --chip AX650 -c 1 --parallel 8
```

使用上述命令编译大语言模型. 关于 `llm_build` 可以参考 [大模型编译文档](https://pulsar2-docs.readthedocs.io/zh-cn/latest/appendix/build_llm.html).