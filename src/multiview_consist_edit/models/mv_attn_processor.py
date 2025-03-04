from diffusers.utils import USE_PEFT_BACKEND
from typing import Callable, Optional
import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class MVXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, weight_matrix=None, attention_op: Optional[Callable] = None):
        if weight_matrix:
            self.bs = weight_matrix.shape[0]
            self.frame_length = weight_matrix.shape[1]
            self.weight_matrix = weight_matrix
        self.attention_op = attention_op

    def update_weight_matrix(self, weight_matrix):
        self.bs = weight_matrix.shape[0]
        self.frame_length = weight_matrix.shape[1]
        self.weight_matrix = weight_matrix    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        garment_fea_attn = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        attn_out = torch.empty_like(query)

        if garment_fea_attn:
            frame_length = self.frame_length + 2 # 2 for two garments
        else:
            frame_length = self.frame_length
        token_num_per_frame = query.shape[1] // frame_length
        # print('000000',query.shape,frame_length)
        heads_num = attn.heads
        for b in range(self.bs):
            for i in range(self.frame_length): 
                curr_q = query[heads_num*b:heads_num*(b+1),token_num_per_frame*i:token_num_per_frame*(i+1),:]
                weight = self.weight_matrix[b,i,:]
                if garment_fea_attn:
                    weight = torch.cat([weight,torch.tensor([1,1],dtype=weight.dtype,device=weight.device)],dim=0)  # garment's attn weight set 1
                weight = weight.repeat_interleave(token_num_per_frame)
                curr_k = key[heads_num*b:heads_num*(b+1)]
                curr_v = value[heads_num*b:heads_num*(b+1)]
                weight = weight.unsqueeze(0).unsqueeze(-1)
                curr_k = weight * curr_k
                hidden_states = xformers.ops.memory_efficient_attention(
                    curr_q, curr_k, curr_v, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                )
                attn_out[heads_num*b:heads_num*(b+1),token_num_per_frame*i:token_num_per_frame*(i+1),:] = hidden_states
        hidden_states = attn_out
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states