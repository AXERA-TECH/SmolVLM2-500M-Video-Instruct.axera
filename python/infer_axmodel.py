from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoConfig
from typing import List, Tuple
from axengine import InferenceSession
from ml_dtypes import bfloat16
from utils.infer_func import InferManager
import argparse
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def run_vision_model(
    encoder,
    pixel_values,
    patch_attention_mask=None,
):
    batch_size = pixel_values.size(0)
    if patch_attention_mask is None:
        patch_size = 16
        patch_attention_mask = torch.ones(
            (
                batch_size,
                pixel_values.size(2) // patch_size,
                pixel_values.size(3) // patch_size,
            )
        )
        patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

    hidden_states = embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

    patch_attention_mask = patch_attention_mask.view(batch_size, -1)
    # The call to `_upad_input` in `_flash_attention_forward` is expensive
    # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
    # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
    if not torch.any(~patch_attention_mask):
        patch_attention_mask = None
    elif not self._use_flash_attention_2:
        patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

    # 保存 vit-encoder 的量化校准集
    # np.save("../model_convert/vit_encoder_calibrations/hidden_states_5.npy", hidden_states.detach().cpu().to(dtype=torch.float32).numpy())
    encoder_outputs = encoder.run(None, {"input": hidden_states.detach().cpu().to(dtype=torch.float32).numpy()})[0]
    encoder_outputs = torch.from_numpy(encoder_outputs).to(device, dtype=hidden_states.dtype)

    return encoder_outputs


def get_image_features(encoder, pixel_values: torch.FloatTensor, pixel_attention_mask: torch.LongTensor = None):
    """
    Encodes images into continuous embeddings that can be forwarded to the language model.

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        pixel_attention_mask (`torch.LongTensor`, *optional*):
            The attention mask indicating padded regions in the image.
    """
    batch_size, num_images, num_channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

    # Remove padding images - padding images are full 0.
    nb_values_per_image = pixel_values.shape[1:].numel()
    real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

    if not any(real_images_inds):
        # no images, leave one empty image.
        real_images_inds[0] = True

    pixel_values = pixel_values[real_images_inds].contiguous()
    # Handle the vision attention mask
    if pixel_attention_mask is None:
        pixel_attention_mask = torch.ones(
            size=[pixel_values.shape[i] for i in (0, 2, 3)],
            dtype=torch.bool,
            device=pixel_values.device,
        )
    else:
        # Remove padding images from the mask
        pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
    patch_size = 16
    patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
    patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
    patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    # Get sequence from the vision encoder
    image_hidden_states = run_vision_model(encoder, pixel_values, patch_attention_mask)

    # Modality projection & resampling
    # image_hidden_states = connector(image_hidden_states) # 已经 fuse 到了 onnx 中
    return image_hidden_states


def inputs_merger(
        input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
    """
    This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
    The merging happens as follows:
    - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
    - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
    We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
    - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
    - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
    """
    _, patch_size, _ = image_hidden_states.shape

    image_mask = input_ids == 49190 # self.image_token_id
    num_image_tokens = image_mask.sum(dim=1)
    if not torch.all(num_image_tokens % patch_size == 0):
        raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

    blocks_per_sample = num_image_tokens // patch_size

    offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
    block_offset = offsets[:-1]
    row_cum = image_mask.cumsum(dim=-1)
    chunk_idx = (row_cum - 1) // patch_size
    local_idx = (row_cum - 1) % patch_size
    block_idx = block_offset.unsqueeze(1) + chunk_idx

    image_embeds = torch.zeros_like(inputs_embeds)
    image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

    merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
    return merged_embeds


if __name__ == "__main__":

    """
    python3 infer_axmodel.py -i ../assets/panda.jpg --vit_model ./vit-models/vision_model.axmodel
    """

    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="./SmolVLM2-500M-Video-Instruct/",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./SmolVLM2-500M-Video-Instruct_axmodel/",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("--vit_model", type=str, default='./vit-models/vision_model.axmodel',
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("-i", "--images", type=str, default="../assets/bee.jpg",
                        help="Path to the test image.")
    parser.add_argument("-q", "--question", type=str, default="Can you describe this image?",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()

    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path
    images = args.images
    prompt = args.question

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.load("SmolVLMVisionEmbeddings.pkl", map_location=device, weights_only=False)
    embeds = np.load(os.path.join(axmodel_path, "model.embed_tokens.weight.npy"))

    encoder = InferenceSession(args.vit_model)

    processor = AutoProcessor.from_pretrained(hf_model_path)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    TARGET_IMAGE_SIZE = (512, 512)
    image = Image.open(images).convert('RGB')

    # 固定输入图像 size: 512x512
    preprocess = Compose([
        Resize(TARGET_IMAGE_SIZE),
        # ToTensor(),
        # Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ])

    preprocessed_image = preprocess(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": preprocessed_image}, # 这里可以直接使用 PIL Image 对象
                # {"type": "image", "url": images}, # 也可以使用 url
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)

    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]
    input_ids = inputs["input_ids"]
    input_ids_length = input_ids.shape[1]

    inputs_embeds = np.take(embeds, input_ids[0].cpu().numpy().tolist(), axis=0)[None, ...]
    inputs_embeds = torch.from_numpy(inputs_embeds).to(device, dtype=torch.bfloat16)

    """
    miniforge-pypy3/envs/lerobot/lib/python3.10/site-packages/transformers/models/smolvlm/modeling_smolvlm.py(681)get_image_features()
    """
    image_hidden_states = get_image_features(encoder, pixel_values, pixel_attention_mask)

    inputs_embeds = inputs_merger(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_hidden_states=image_hidden_states,
    ).to(dtype=torch.float32).cpu().numpy()

    prefill_data = inputs_embeds
    prefill_data = prefill_data.astype(bfloat16)
    token_ids = input_ids[0].cpu().numpy().tolist()
    token_len = len(token_ids)
    cfg = config.text_config

    imer = InferManager(cfg, axmodel_path)

    token_ids = imer.prefill(tokenizer, token_ids, prefill_data[0], slice_len=128)
    imer.decode(tokenizer, token_ids, embeds, slice_len=128)
    print("\n")
