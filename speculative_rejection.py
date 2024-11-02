
from generator import Generator

def get_output_texts(
    generation_ids: torch.LongTensor,
    prompt: str,
    generation_tokenizer,
    skip_special_tokens: bool = False,
) -> list[str]:
    generation_texts = generation_tokenizer.batch_decode(
        generation_ids, skip_special_tokens=skip_special_tokens
    )
    output_texts: list[str] = []
    for generation_text in generation_texts:
        generation_text = generation_text.replace(
            "<s> [INST]", "<s>[INST]"
        )  # for llama-2-chat-hf
        split_pieces = generation_text.split(prompt)
        # print(generation_ids)
        # print(generation_tokenizer.decode(generation_ids[0]))
        # print(prompt)
        # print(generation_text)
        # # write to txt:
        # with open('output.txt', 'w') as f:
        #     f.write(generation_text)
        # with open('output2.txt', 'w') as f:
        #     f.write(prompt)
        try:
            assert (
                prompt in generation_text
            ), f"prompt: {prompt} | generation_text: {generation_text}"
            assert (
                len(split_pieces) > 1
            ), f"prompt: {prompt} | generation_text: {generation_text}, {len(split_pieces)}, {split_pieces}"
            output_text = prompt.join(split_pieces[1:])
        except:
            output_text = generation_text[len(prompt) :]
        output_texts.append(output_text)
    return output_texts
def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding

def unpad_output_texts(output_texts: list[str], stop_tokens: list[str]) -> list[str]:
    unpadded_texts: list[str] = []
    for output_text in output_texts:
        for stop_token in stop_tokens:
            output_text = output_text.split(stop_token)[0]
        unpadded_texts.append(output_text)
    return unpadded_texts
  
@torch.inference_mode()
def get_memory_constrained_generation(
    generation_model: transformers.LlamaForCausalLM,
    generation_ids: torch.LongTensor,
    terminators: list[int | None],
    pad_token_id: int | None,
    args,
) -> torch.LongTensor:

    past_key_values = None
    batch_size = generation_ids.shape[0]
    finished_generations = torch.zeros(batch_size).bool().to(generation_model.device)
    while generation_ids.shape[-1] < args.max_tokens:
        try:
            out_dict = generation_model.generate(
                generation_ids,
                pad_token_id=pad_token_id,
                max_new_tokens=1,
                eos_token_id=terminators,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
            )
            if "past_key_values" in out_dict:
                past_key_values = out_dict.past_key_values
            else:
                raise Exception("past_key_values (KV cache) not found in model output")
            generation_ids = out_dict.sequences
        except torch.cuda.OutOfMemoryError:
            break
        just_finished = generation_ids[:, -1] == pad_token_id
        finished_generations = finished_generations | just_finished
        if torch.all(finished_generations):
            break
    return generation_ids



def get_templated_prompt(
    prompt: str,
    llm_name: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> str:
    if "Instruct" in llm_name:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    elif any(s in llm_name for s in ["sft10k", "alpaca-7b", "dpo", "ppo", "human"]):
        templated_prompt = f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    elif "llama-2" in llm_name.lower():
        templated_prompt = f"<s>[INST]\n{prompt} [/INST]"
    else:
        templated_prompt = generation_tokenizer.bos_token + prompt
    return templated_prompt


def write_to_disk(
    all_data: list[dict[str, Any]],
    output_folder: str,
    initial_memory: int,
    pretty_print_output: bool = False,
    record_memory: bool = False,
    force_dump: bool = False,
) -> None:
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    prompt_idx: int = (
        all_data[0]["prompt"]["JSON_idx"]
        if "prompt" in all_data[0]
        and type(all_data[0]["prompt"]) == dict
        and "JSON_idx" in all_data[0]["prompt"]
        else 0
    )
    llm_name: str = all_data[0]["llm_name"]
    reward_model_name: str = all_data[0]["reward_model_name"]
    write_filename = f"{llm_name}_{reward_model_name}_prompt_{prompt_idx:04d}.json"
    write_path = os.path.join(output_folder, write_filename)
    if force_dump or (record_memory and prompt_idx == 0):
        dump_memory_snapshot(write_path, initial_memory)
    if force_dump:
        return
    print_best_trajectory(all_data)
    with open(write_path, "w") as fp:
        if pretty_print_output:
            json.dump(all_data, fp, indent=4)
        else:
            json.dump(all_data, fp)
        print(f"Wrote data to {write_filename}")

def compute_scores(
    question: str,
    output_texts: list[str],
    reward_model_name: str,
    reward_tokenizer,
    reward_model,
) -> list[float]:
    reward_tokens = get_reward_tokens(
        question,
        output_texts,
        reward_model_name,
        reward_tokenizer,
        reward_model.device,
    )
    # print(f"reward_tokens: {reward_tokens}")
    reward_list = get_rewards(reward_model_name, reward_model, reward_tokens)

    if reward_list is None:
        raise Exception("Could not compute scores...")
    return reward_list
def get_memory_constrained_batch_size(length: int, llm_name: str) -> int:
    a, b = get_inverse_function_params(llm_name)
    return int(a / (length + b))


def get_inverse_function_params(llm_name: str) -> tuple[float, float]:
    # NOTE: these parameters are computed by fitting an inverse function to data
    # generated by benchmark_batch_size.py
    if llm_name == "sft10k" or llm_name == "alpaca-7b":
        return (53288.568, 9.164)
    elif llm_name == "Meta-Llama-3-8B":
        return (61626.403, 2.076)
    elif llm_name == "Meta-Llama-3-8B-Instruct" or "Mistral-7B" in llm_name:
        return (61562.069, 2.058)
    else:
        raise Exception("Unknown LLM name")from utils.trajectory import Trajectory

def validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 1.0):
        raise Exception("args.alpha expected to be in [0.0, 1.0)")
import torch, gc
from engine.models.llm import LLM


class SpeculativeRejection(Generator):
    def generate(self, prompt: str, prompt_dict: dict | None = None) -> None:
        if prompt_dict is None:
            prompt_dict = prompt
        self.prepare_generation(prompt_dict)
        self.clock.reset()
        self.clock.start()
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )
        alpha: float = self.args.alpha
        validate_alpha(alpha)
        batch_encoding = get_input_encoding(
            [self.templated_prompt],
            self.generation_model,
            self.generation_tokenizer,
        )
        input_length = batch_encoding.input_ids.shape[-1]
        batch_size = get_memory_constrained_batch_size(input_length, self.args.llm_name)

        # set max tokens for engine
        max_all_tokens = min(
            self.args.max_tokens, self.args.max_gen_tokens + input_length
        )
        # decide init bsz for engine
        if isinstance(self.generation_model, LLM):
            self.generation_model.max_tokens = max_all_tokens
            batch_size = min(int(batch_size * 2), 1000)
            self.generation_model.tokenizer = self.generation_tokenizer

            while True:
                gen_len = self.generation_model.get_gen_len(
                    batch_size=batch_size, cur_len=input_length
                )
                if gen_len >= 8:
                    break
                batch_size = int(batch_size * 0.9)

        current_generations = [self.templated_prompt] * batch_size
        self.clock.stop("hyperparameter selection")
        print(f"input_length: {input_length}")
        self.clock.start()
        current_length = input_length

        while current_length < max_all_tokens:
            if isinstance(self.generation_model, LLM):
                batch_encoding = self.generation_model.batch_encode(current_generations)
            else:
                batch_encoding = get_input_encoding(
                    current_generations,
                    self.generation_model,
                    self.generation_tokenizer,
                )
            self.clock.stop("tokenization")
            self.clock.start()
            try:
                if isinstance(self.generation_model, LLM):
                    batch_size = batch_encoding.shape[0]
                    cur_len = batch_encoding.shape[1]
                    gen_len = self.generation_model.get_gen_len(
                        batch_size=batch_size, cur_len=cur_len
                    )
                    if gen_len < 1:
                        gen_len = 1
                    assert gen_len > 0
                    partial_generation = self.generation_model.generate(
                        input_ids=batch_encoding,
                        batch_size=batch_size,
                        gen_len=gen_len,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        temperature=self.args.temperature,
                    )
                else:
                    partial_generation = get_memory_constrained_generation(
                        self.generation_model,
                        batch_encoding.input_ids,
                        self.terminators,
                        self.generation_tokenizer.pad_token_id,
                        self.args,
                    )
            except Exception as e:
                print(e)
                write_to_disk(
                    self.all_data,
                    "./output_crashes",
                    self.initial_memory,
                    self.args.pretty_print_output,
                    self.args.record_memory,
                    force_dump=True,
                )
                raise Exception("Memory error occurred during generation")
            current_length = partial_generation.shape[-1]
            self.clock.stop(
                f"generation - partial_generation.shape {partial_generation.shape}"
            )
            print(f"partial_generation shape: {partial_generation.shape}")

            self.clock.start()
            padded_output_texts = get_output_texts(
                    partial_generation,
                    self.templated_prompt,
                    self.generation_tokenizer,
                    skip_special_tokens=False,
                )
            unpadded_output_texts = unpad_output_texts(
                    padded_output_texts, self.stop_tokens
                )
            self.clock.stop(f"decoding - current_length {current_length}")
            
            if self.is_self_reward:
                reward_list = self.generation_model.self_evaluate(partial_generation)
            else:
                self.clock.start()
                reward_list = compute_scores(
                    prompt,
                    unpadded_output_texts,
                    self.reward_model_name,
                    self.reward_tokenizer,
                    self.reward_model,
                )
                self.clock.stop(f"reward - current_length {current_length}")
            
            self.clock.start()
            current_trajectories: list[Trajectory] = [
                Trajectory(
                    self.prompt,
                    self.templated_prompt,
                    padded_output_text,
                    unpadded_output_text,
                    score,
                )
                for padded_output_text, unpadded_output_text, score in zip(
                    padded_output_texts, unpadded_output_texts, reward_list
                )
            ]
            current_generations = self.perform_speculative_rejection(
                current_trajectories, alpha
            )
            if len(current_generations) == 0:
                break
            self.clock.stop(f"speculative rejection - current_length {current_length}")
            self.clock.start()
        self.trajectories = (
            self.trajectories + current_trajectories + self.finished_trajectories
        )
        self.clock.stop("finish")
        self.post_generation()

    def perform_speculative_rejection(
        self,
        current_trajectories: list[Trajectory],
        alpha: float,
    ) -> list[str]:
        previous_finished_trajectories = [
            trajectory for trajectory in self.trajectories if trajectory.finished
        ]
        self.finished_trajectories += previous_finished_trajectories
        trajectories_to_rank = previous_finished_trajectories + current_trajectories
        trajectories_to_rank.sort(key=lambda trajectory: trajectory.score, reverse=True)
        keep_fraction = 1.0 - alpha
        keep_amount = int(round(keep_fraction * len(trajectories_to_rank)))
        self.trajectories = trajectories_to_rank[:keep_amount]
        generating_trajectories = [
            trajectory for trajectory in self.trajectories if not trajectory.finished
        ]
        current_generations = [
            trajectory.templated_prompt + trajectory.unpadded_output_text
            for trajectory in generating_trajectories
        ]
        return current_generations

class LLM:
    def __init__(
        self,
        model_name: str = "hmomin/sft10k",
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    ) -> None:

        self.local_files_only = local_files_only
        print(f"Initializing LLM with {model_name} on {device} with dtype: {dtype}")
        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, legacy=False, local_files_only=local_files_only
        )
        self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.softmax_scale = 1 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=self.dtype, device=self.device)
        )

        torch.cuda.synchronize()
        model_hbm = gpu_memory(self.device)
        print(f"[model w/o cache init on {device}]:  {model_hbm}")

        self.prefill_phase = False

    def __str__(self) -> str:
        return f"LLM: {self.model_name}, device: {self.device}, dtype: {self.dtype}"

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(
            self.config.max_position_embeddings,
            device=self.device,
            dtype=inv_freq.dtype,
        )
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        )
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        try:
            self.cos_cache = (
                hf_model.model.layers[0]
                .self_attn.rotary_emb.cos_cached.to(self.device)
                .to(self.dtype)
            )
            self.sin_cache = (
                hf_model.model.layers[0]
                .self_attn.rotary_emb.sin_cached.to(self.device)
                .to(self.dtype)
            )
        except:
            print("RoPE cache not found, initializing RoPE cache")
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache(
                hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.to(self.device)
            )

            # for vllm
            self.cos_sin_cache = torch.cat(
                (self.cos_cache[:, :64], self.sin_cache[:, :64]), dim=-1
            )

        self.layers: list[LlamaLayer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: LlamaLayer,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        bsz = hidden_states.shape[0]
        hidden_states = layer_norm(
            hidden_states,
            buffer.input_layernorm_variance_epsilon,
            buffer.input_layernorm_weight,
        )
        qkv = F.linear(hidden_states, buffer.wqkv)
        query_states, key_states, value_states = qkv.split(
            [buffer.q_size, buffer.kv_size, buffer.kv_size], dim=-1
        )
        return (
            query_states,
            key_states,
            value_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2),
        )

    def post_attention_compute(
        self, attn_output: torch.Tensor, residual: torch.Tensor, buffer: LlamaLayer
    ):
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(
            hidden_states,
            buffer.post_attention_layernorm_variance_epsilon,
            buffer.post_attention_layernorm_weight,
        )

        hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
        d = hidden_states.shape[-1] // 2
        output_shape = hidden_states.shape[:-1] + (d,)
        out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        vllm._custom_ops.silu_and_mul(out, hidden_states)

        hidden_states = F.linear(out, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def layer_compute_wo_cache(
        self,
        buffer: LlamaLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
    ):
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )

        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states.transpose(1, 2),
            v_cache=value_states.transpose(1, 2),
            causal=True,
        )

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

        hidden_states = self.post_attention_compute(
            hidden_states,
            residual,
            buffer,
        )

        return hidden_states

    @torch.inference_mode()
    def layer_compute(
        self,
        buffer: LlamaLayer,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        storage_ids: torch.LongTensor,
    ):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, position_ids
        )
        key_states, value_states = self.kv_cache.update_kv_cache(
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            layer_idx,
            storage_ids,
        )

        hidden_states = flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states,
            v_cache=value_states,
            cache_seqlens=self.kv_cache.kv_offset,
            causal=True,
        )

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

        hidden_states = self.post_attention_compute(
            hidden_states,
            residual,
            buffer,
        )

        return hidden_states

    @torch.inference_mode()
    def inference(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        storage_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        return_logits=True,
    ):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(
                self.layers[idx],
                idx,
                hidden_states,
                position_ids,
                attention_mask,
                storage_ids,
            )

        hidden_states = layer_norm(
            hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon
        )

        if self.prefill_phase:  # prefill
            hidden_states = hidden_states[:, -1:, :]

        if return_logits:
            logits = F.linear(hidden_states, self.lm_head).float()
            return logits

    @torch.inference_mode()
    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        vllm._custom_ops.rotary_embedding(
            position_ids, q, k, 128, self.cos_sin_cache, True
        )
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(
            input_len, dtype=torch.long, device=self.device
        ) + past_len.unsqueeze(1)
        storage_ids = position_ids.clone()
        return position_ids, storage_ids

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()

        iter_prefill = max((input_ids.shape[0] * input_ids.shape[1]) // (5000), 1)
        T = (input_ids.shape[1] + iter_prefill - 1) // iter_prefill
        iter_prefill = (input_ids.shape[1] + T - 1) // T
        for i in range(iter_prefill):
            if input_ids[:, i * T : (i + 1) * T].shape[-1] < 1:
                print("break")
                break
            try:
                position_ids, storage_ids = self.get_ctx(
                    input_ids[:, i * T : (i + 1) * T]
                )
                if i == iter_prefill - 1:
                    logits = self.inference(
                        input_ids=input_ids[:, i * T : (i + 1) * T],
                        position_ids=position_ids,
                        attention_mask=None,
                        storage_ids=storage_ids,
                    )
                else:
                    self.inference(
                        input_ids=input_ids[:, i * T : (i + 1) * T],
                        position_ids=position_ids,
                        attention_mask=None,
                        storage_ids=storage_ids,
                        return_logits=False,
                    )
            except Exception as e:
                print(
                    position_ids,
                    storage_ids,
                    input_ids.shape,
                    i,
                    T,
                    input_ids[:, i * T : (i + 1) * T].shape,
                    self.kv_cache.kv_offset,
                )
                raise e

        return logits

    def encode(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return input_ids

    def batch_encode(self, text_list):
        input_ids = self.tokenizer(
            text_list, return_tensors="pt", padding=True, add_special_tokens=False
        ).input_ids.to(self.device)

        min_length = min(
            input_ids.shape[1],
            torch.min(
                torch.sum(input_ids != self.tokenizer.pad_token_id, dim=1)
            ).item(),
        )
        input_ids = input_ids[:, :min_length]
        return input_ids

    def decode(self, input_ids: torch.LongTensor):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def init_kv_cache(self, max_length: int = 256):
        if (
            "TinyLlama" in self.model_name
            or "JackFram" in self.model_name
            or "68" in self.model_name
        ):
            if not hasattr(self, "gamma"):
                self.gamma = 4
            self.kv_cache = SinkCache(
                self.config,
                max_length=max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                gamma=self.gamma,
            )
        else:
            self.kv_cache = KV_Cache(
                self.config,
                max_length=max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
            )
        torch.cuda.synchronize()
        model_kv_cache_hbm = gpu_memory(self.device)
        # print(self.kv_cache)
        print(
            f"[model ({self.model_name}) w/ cache init on {self.device}]:  {model_kv_cache_hbm}"
        )

    def get_gen_len(self, batch_size: int, cur_len: int):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem_GB = gpu_free_memory(self.device) - 15  # 4GB buffer
        # for example, (1, 32K) seq len = 16GB for sft10k
        max_seq = int(free_mem_GB * 1024 * 2 / batch_size * (self.num_key_value_groups))
        max_seq = min(max_seq, self.max_tokens)
        # assert max_seq > cur_len, f"Max sequence length {max_seq} is less than input length {cur_len}"
        return max_seq - cur_len

    def get_batch_size(self, max_seq: int):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem_GB = gpu_free_memory(self.device) - 4
        batch_size = int(free_mem_GB * 1024 * 2 / max_seq * (self.num_key_value_groups))
        return batch_size


    @torch.inference_mode()
    def inference_wo_cache(self, input_ids: torch.LongTensor):
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        input_len = input_ids.shape[-1]
        position_ids = torch.arange(
            input_len, dtype=torch.long, device=self.device
        )

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute_wo_cache(
                self.layers[idx],
                idx,
                hidden_states,
                position_ids,
            )

        hidden_states = layer_norm(
            hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon
        )

        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def self_evaluate(self, input_ids: torch.LongTensor):
        score = []
        
        T = 1
        num_iter = (input_ids.shape[0] + T - 1) // T

        for i in range(num_iter):
            input_ = input_ids[i * T : (i + 1) * T]
            logits = self.inference_wo_cache(input_)

            loss = None

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # [T, seq, vocab_size]
            shift_labels = input_ids[i * T : (i + 1) * T, 1:].contiguous() # [T, seq]
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Enable model parallelism
            loss = -torch.exp(loss_fct(shift_logits, shift_labels).view(T, -1).mean(dim=-1))
            score.append(loss.tolist())
        
        return score

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1024,
        gen_len: int = 256,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        benchmark: bool = False,
    ):

        self.batch_size = batch_size

        if type(input_ids) == str:
            input_ids = self.encode(input_ids)

        if input_ids.size(0) != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: {input_ids.size(0)} != {self.batch_size}"
            )

        if benchmark:
            torch.cuda.synchronize()
            start = time.time()

        # init kv cache
        max_length = input_ids.size(1) + gen_len
        self.init_kv_cache(max_length=max_length)

        # prefill
        self.prefill_phase = True
        logits = self.prefill(input_ids)
        self.prefill_phase = False

        next_token = sample(
            norm_logits(
                logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k
            )
        )

        if benchmark:
            torch.cuda.synchronize()
            prefill = time.time()

        n = 0

        generated_ids = [[] for _ in range(self.batch_size)]
        for i, token in enumerate(next_token.tolist()):
            generated_ids[i].extend(input_ids[i].tolist())
            generated_ids[i].extend(token)

        finished_generations = torch.zeros(batch_size).bool().to(self.device)
        while n < gen_len:
            position_ids, storage_ids = self.get_ctx(next_token)
            attention_mask = None

            logits = self.inference(
                input_ids=next_token,
                position_ids=position_ids,
                attention_mask=attention_mask,
                storage_ids=storage_ids,
            )
            next_token = sample(
                norm_logits(
                    logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k
                )
            )

            n += 1
            for i, token in enumerate(next_token.tolist()):
                generated_ids[i].extend(token)

            just_finished = (
                torch.LongTensor(np.array(generated_ids))[:, -1].to(self.device)
                == self.tokenizer.pad_token_id
            )
            finished_generations = finished_generations | just_finished
            if torch.all(finished_generations):
                break

        if benchmark:
            torch.cuda.synchronize()
            end = time.time()
            print(
                f"Time taken: {end-prefill} to generate {gen_len} tokens, Prefill time: {prefill-start} ({input_ids.shape})"
            )

        # free KV Cache
        del self.kv_cache
        self.kv_cache = None
        gc.collect()
        torch.cuda.empty_cache()

        return torch.LongTensor(np.array(generated_ids)).to(self.device)
