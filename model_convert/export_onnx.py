import torch
import torch.nn as nn
import argparse
from loguru import logger
import os
import onnx
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import numpy as np
# from transformers import AutoModelForCausalLM
from typing import List, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText


def onnx_sim(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    logger.info(f"onnx simpilfy successed, and model saved in {onnx_path}")



if __name__ == '__main__':
    
    """
    Usage:
        python3 export_onnx.py -m /path/your/hugging_face/models/SmolVLM2-500M-Video-Instruct/ -o ./vit-models
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", type=str, help="hugging fance model path")
    parser.add_argument("-o", "--onnx_save_dir", type=str, default='./vit-models', help="vit encoder onnx save path")
    args = parser.parse_args()

    model_path = args.model
    onnx_save_dir = args.onnx_save_dir

    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)
    
    # model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"
    ).to("cuda")

    vision_model = model.model.vision_model
    connector = model.model.connector

    ################# EXPORT VISION MODEL ######################
    vision_model_onnx_save_dir = os.path.join(
        onnx_save_dir,
        'vision_model.onnx'
    )

    class Warpped_VISION_MODEL(nn.Module):
        def __init__(self, vision_model, connector):
            super().__init__()
            self.encoder = vision_model.encoder
            self.post_layernorm = vision_model.post_layernorm
            self.connector = connector

        def forward(self, inputs_embeds) -> torch.Tensor:
            encoder_outputs = self.encoder(inputs_embeds=inputs_embeds)
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)
            last_hidden_state = self.connector(last_hidden_state)
            return last_hidden_state
    
    warpped_vision_model = Warpped_VISION_MODEL(vision_model, connector).to(device)
    warpped_vision_model.eval()

    pixel_values_embeds = torch.randn(17, 1024, 768).to(device, dtype=torch.float32)
    
    torch.onnx.export(
        warpped_vision_model.to(dtype=torch.float32),
        pixel_values_embeds,
        vision_model_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=True,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
    )
    logger.debug("export vision_model onnx succee!")
    onnx_sim(vision_model_onnx_save_dir)
