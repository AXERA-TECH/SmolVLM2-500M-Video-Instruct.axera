import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from axengine import InferenceSession
from ml_dtypes import bfloat16


class InferManager:
    def __init__(self, config, model_dir):

        self.config = config
        self.max_seq_len = 2559
        self.kv_dim = config.hidden_size // config.num_attention_heads * config.num_key_value_heads

        self.k_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]

        self.decoder_sessions = []
        for layer_idx in tqdm(range(config.num_hidden_layers), desc="Init InferenceSession"):
            session = InferenceSession(
                f"{model_dir}/llama_p128_l{layer_idx}_together.axmodel"
            )
            self.decoder_sessions.append(session)
        self.post_process_session = InferenceSession(
            f"{model_dir}/llama_post.axmodel"
        )
        print("Model loaded successfully!")

    @staticmethod
    def _top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)
        filtered = probs.copy()
        cumulative = 0
        for idx in sorted_indices[::-1]:
            if cumulative >= p:
                filtered[idx] = 0
            cumulative += filtered[idx]
        return filtered / cumulative

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / np.sum(exp_logits)).astype(np.float64)

    def post_process(self, logits, top_k=1, top_p=0.9, temperature=0.6):
        logits = logits.astype(np.float32).flatten()
        candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[candidate_indices] / temperature
        candidate_probs = self._softmax(candidate_logits)
        candidate_probs = self._top_p(candidate_probs, top_p)
        candidate_probs = candidate_probs.astype(np.float64) / candidate_probs.sum()
        chosen_idx = np.random.multinomial(1, candidate_probs).argmax()
        next_token = candidate_indices[chosen_idx]
        return next_token, candidate_indices, candidate_probs

    def gen_slice_indices(self, token_len, prefill=128, expand=128):
        remaining = max(0, token_len - prefill)
        extra_blocks = (remaining + expand - 1) // expand
        return list(range(extra_blocks + 1))

    def prefill(
        self,
        tokenizer,
        token_ids,
        embed_data,
        slice_len=128,
    ):
        """
        Prefill step for chunked inference.
        """
        seq_len = len(token_ids)
        slice_indices = [i for i in range(seq_len // slice_len + 1)]
        print(f"slice_indices: {slice_indices}")
        # total_prefill_len = (
        #     slice_len * slice_indices[-1]
        #     if slice_indices[-1] != 0
        #     else slice_len
        # )
        total_prefill_len = slice_len * (slice_indices[-1] + 1)
        # slice_indices = self.gen_slice_indices(seq_len)
        # import pdb; pdb.set_trace()

        if total_prefill_len > 0:
            for slice_idx in slice_indices:
                indices = np.arange(
                    slice_idx * slice_len,
                    (slice_idx + 1) * slice_len,
                    dtype=np.uint32
                ).reshape((1, slice_len))

                mask = (
                    np.zeros((1, slice_len, slice_len * (slice_idx + 1)))
                    - 65536
                )
                data = np.zeros((1, slice_len, self.config.hidden_size)).astype(bfloat16)
                for i, t in enumerate(
                    range(
                        slice_idx * slice_len,
                        (slice_idx + 1) * slice_len,
                    )
                ):
                    if t < len(token_ids):
                        mask[:, i, : slice_idx * slice_len + i + 1] = 0
                        data[:, i : i + 1, :] = (
                            embed_data[t]
                            .reshape((1, 1, self.config.hidden_size))
                            .astype(bfloat16)
                        )

                remain_len = (
                    seq_len - slice_idx * slice_len
                    if slice_idx == slice_indices[-1]
                    else slice_len
                )
                mask = mask.astype(bfloat16)
                for layer_idx in range(self.config.num_hidden_layers):
                    input_feed = {
                        "K_cache": (
                            self.k_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "V_cache": (
                            self.v_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    # import pdb; pdb.set_trace()
                    outputs = self.decoder_sessions[layer_idx].run(None, input_feed, shape_group=slice_idx + 1)
                    self.k_caches[layer_idx][
                        :,
                        slice_idx * slice_len : slice_idx * slice_len + remain_len,
                        :,
                    ] = outputs[0][:, :remain_len, :]
                    self.v_caches[layer_idx][
                        :,
                        slice_idx * slice_len : slice_idx * slice_len + remain_len,
                        :,
                    ] = outputs[1][:, :remain_len, :]
                    data = outputs[2]

                print("Slice prefill done:", slice_idx)
            post_out = self.post_process_session.run(
                None,
                {
                    "input": data[
                        :, seq_len - (len(slice_indices) - 1) * slice_len - 1, None, :
                    ]
                }
            )[0]
            next_token, possible_tokens, possible_probs = self.post_process(post_out)
            possible_decoded = [tokenizer.decode([t]) for t in possible_tokens]
            possible_probs_str = [str((t, p)) for t, p in zip(possible_decoded, possible_probs)]
            token_ids.append(next_token)
            return token_ids

    def decode(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        slice_len=128
    ):
        # import pdb; pdb.set_trace()
        print("answer >>", tokenizer.decode(token_ids[-1], skip_special_tokens=True), end='', flush=True)
        self.max_seq_len = 2559
        mask = np.zeros((1, 1, self.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if prefill_len > 0:
            mask[:, :, :seq_len] = 0
        for step_idx in range(self.max_seq_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue
            # import pdb; pdb.set_trace()
            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = embed_matrix[cur_token, :].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)
            for layer_idx in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[layer_idx],
                    "V_cache": self.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.decoder_sessions[layer_idx].run(None, input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx, :] = outputs[0][:, :, :]
                self.v_caches[layer_idx][:, step_idx, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue
            else:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, possible_tokens, possible_probs = self.post_process(post_out)
                token_ids.append(next_token)
                if next_token == tokenizer.eos_token_id and next_token > seq_len:
                    break
            print(tokenizer.decode(next_token, skip_special_tokens=True), end='', flush=True)

