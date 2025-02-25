from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union, Iterable

if TYPE_CHECKING:
    from transformers import Qwen2Config
else:
    Qwen2Config = None
from mindspore import Parameter, Tensor, mint, nn, jit, mutable
from mindspore.common import dtype as mstype


from vllm_mindspore.attention import Attention

from vllm_mindspore.model_executor.layers.activation import SwiGLU
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import \
    LogitsProcessor
from vllm.model_executor.layers.quantization import \
    QuantizationConfig
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.layers.sampler import (SamplerOutput,
                                                          get_sampler)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import \
    default_weight_loader
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
from vllm_mindspore.model_executor.models.model_base import MsModelBase


from vllm.config import CacheConfig, VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.attention.backends.abstract import AttentionType
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.attention.backends.abstract import AttentionMetadata


class Qwen2MLP(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            params_dtype=mstype.bfloat16
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            params_dtype=mstype.bfloat16
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SwiGLU()

    @jit
    def construct(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Cell):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position: int = 4096 * 32,
            rope_theta: float = 10000,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            rope_scaling: Optional[Tuple] = None,
            prefix: str = "",
            attn_type: str = AttentionType.DECODER
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=mstype.bfloat16,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            params_dtype=mstype.bfloat16,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dtype=mstype.bfloat16,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type
        )

    @jit
    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        kv_cache: Tuple[Tensor, Tensor],
        # attn_metadata: AttentionMetadata,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: Tensor,
        batch_valid_length: Tuple[int],
        context_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = mint.split(qkv, (self.q_size, self.kv_size, self.kv_size), -1)
        q, k = self.rotary_emb(positions, q, k, context_lens, num_prefill_tokens)
        attn_output = self.attn(q, k, v, kv_cache, num_prefill_tokens, num_decode_tokens,
                                slot_mapping, batch_valid_length, context_lens, block_tables)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Cell):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       params_dtype=mstype.bfloat16,)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                params_dtype=mstype.bfloat16,)

    @jit
    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        kv_cache: Tuple[Tensor, Tensor],
        # attn_metadata: AttentionMetadata,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: Tensor,
        batch_valid_length: Tuple[int],
        context_lens: Tensor,
        block_tables: Tensor,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            kv_cache,
            num_prefill_tokens,
            num_decode_tokens,
            slot_mapping,
            batch_valid_length,
            context_lens,
            block_tables
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Cell):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                params_dtype=mstype.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                params_dtype=mstype.bfloat16,)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    @jit
    def construct(
        self,
        input_ids: Optional[Tensor],
        positions: Tensor,
        kv_caches: List[Tuple[Tensor, Tensor]],
        # attn_metadata: AttentionMetadata,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: Tensor,
        batch_valid_length: Tuple[int],
        context_lens: Tensor,
        block_tables: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):  # PP 并行对层进行切分
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                num_prefill_tokens,
                num_decode_tokens,
                slot_mapping,
                batch_valid_length,
                context_lens,
                block_tables,
                residual
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]], params_dict: Dict[str, Parameter]):
        loaded_params: Set[str] = set()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class Qwen2ForCausalLM(MsModelBase):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              params_dtype=mstype.bfloat16,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.set_model_inputs()

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tuple[Tensor, Tensor]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: IntermediateTensors = None,
        inputs_embeds: Tensor = None
    ) -> Union[Tensor, IntermediateTensors]:
        if attn_metadata.num_prefill_tokens > 0:
            input_ids = input_ids.expand_dims(0)
        if attn_metadata.num_decode_tokens > 0:
            input_ids = input_ids.expand_dims(1)
        model_output = self.model(input_ids,
                                  positions,
                                  kv_caches,
                                  **dict(attn_metadata),
                                  intermediate_tensors=intermediate_tensors,
                                  inputs_embeds=inputs_embeds)
        if attn_metadata.num_prefill_tokens > 0:
            model_output = model_output.squeeze(0)
        if attn_metadata.num_decode_tokens > 0:
            model_output = model_output.squeeze(1)
        return model_output

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        params_dict = self.parameters_dict()
        self.model.load_weights(weights, params_dict)

    def sample(
        self, logits: Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits
