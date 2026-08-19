
from torch import Tensor
from einops import einsum, rearrange
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, boolean_mask: Tensor | None = None):

    *_, seq_len, d_k = query.shape

    qk = einsum(
        query,
        key,
        "batch_size ... seq_len1 d_k, batch_size ... seq_len2 d_k -> batch_size ... seq_len1 seq_len2",
    )

    logits = qk / (d_k**0.5)

    if boolean_mask is not None:
        # Boolean_mask should have shape "batch_size ... seq_len seq_len"
        logits = logits.masked_fill(boolean_mask, float("-inf"))

    attn_weights = softmax(logits, dim=-1)
    output = einsum(
        attn_weights,
        value,
        "batch_size ... seq_len1 seq_len2, batch_size ... seq_len2 d_v -> batch_size ... seq_len1 d_v",
    )

    # Return a tensor of shape "batch_size ... seq_len d_v"
    return output

def compute_better_attention(self, query, key, value, dim, use_mask, use_rope):
    ######################
    #### Input Shapes ####
    ######################
    # query: (bs,l1,l2,d1)
    # key: (bs,l1,l2,d1)
    # value: (bs,l1,l2,nh,dh)
    # dim: mode_index
    # mask: (bs,nh,l1,l2)
    # d_head = d1/nh
    ######################

    l = query.shape[dim]
    q = query.transpose(dim, -2)  # (bs ... l1 d1)
    k = key.transpose(dim, -2)  # (bs ... l2 d1)
    v = value.transpose(dim, -3)  # (bs ... l1 nh dh)

    q = q.unflatten(dim=-1, sizes=(self.n_head, self.d_head))  # (bs ... l1 nh dh)
    k = k.unflatten(dim=-1, sizes=(self.n_head, self.d_head))  # (bs ... l2 nh dh)

    def pool(x):
        return einsum(x, "bs ... l nh dh -> bs l nh dh")

    q = self.q_norm(pool(q))  # (bs l1 nh dh)
    k = self.k_norm(pool(k))  # (bs l2 nh dh)

    #################### RoPE ##################
    if use_rope and dim in self.rope_dims:
        rotary_emb = self.ropes[f"dim_{dim}"]

        if self.token_positions == None:
            (bs, l, nh, dh) = q.shape
            self.token_positions = torch.empty(
                size=(bs, nh, l), device=self.factory_kwargs["device"]
            )
            self.token_positions[:] = torch.arange(
                l, device=self.factory_kwargs["device"]
            )

        q = rearrange(q, "bs l nh dh->bs nh l dh")
        k = rearrange(k, "bs l nh dh->bs nh l dh")

        # RoPE exige x cada um: [..., nh, l, d_k]
        # RoPE exige token_positions cada um: [..., l]

        q = rotary_emb(x=q, token_positions=self.token_positions)
        k = rotary_emb(x=k, token_positions=self.token_positions)

        q = rearrange(q, "bs nh l dh->bs l nh dh")
        k = rearrange(k, "bs nh l dh->bs l nh dh")
    ############################################

    att = einsum(q, k, "bs l1 nh d, bs l2 nh d -> bs l1 l2 nh") / math.sqrt(
        q.shape[3]
    )

    #################### MASK ##################
    if use_mask:
        # mask.shape = (bs,nh,n,n)
        bs, *_ = q.shape
        mask = torch.triu(
            torch.ones(
                (bs, self.n_head, self.N[dim - 1], self.N[dim - 1]),
                dtype=torch.bool,
                device=self.factory_kwargs["device"],
            ),
            diagonal=1,
        )
        # Boolean_mask should have shape "batch_size ... l1 l2"
        att = rearrange(att, "bs l1 l2 nh -> bs nh l1 l2")
        att = att.masked_fill(mask, float("-inf"))
        att = rearrange(att, "bs nh l1 l2 -> bs l1 l2 nh")
    ############################################

    att = self.att_dropout(F.softmax(att, dim=2)).float()
    h = einsum(att, v, "bs l1 l2 nh, bs ... l1 nh d -> bs ... l2 nh d")
    return h.transpose(dim, -3), att

def compute_kron_attention(self, query, key, value, dim_target,n_head, mask, positional_encoding):
        ######################
        #### Input Shapes ####
        ######################
        # k is the number of dimensions (k-mode)
        # query: (bs, lq_1, ..., lq_k, d_kron)
        # key: (bs, lk_1, ..., lk_k, d_kron)
        # value: (bs,lv_1, ..., lv_k,nh,d_head)
        # dim: mode_index
        # mask: (bs, nh, lq_target, lk_target)
        # d_head = d_kron/nh
        ######################

        d_head=query.shape[-1]//n_head
        q = query.transpose(dim_target, -2)  # shape = (bs ... lq_target d_kron)
        k = key.transpose(dim_target, -2)  # shape = (bs ... lk_target d_kron)
        v = value.transpose(dim_target, -3)  # shape = (bs ... lv_target n_head d_head)

        q = rearrange(q, 'bs ... (n_head d_head) -> bs ... n_head d_head', n_head=n_head)  # (bs ... lq_target n_head d_head)
        k = rearrange(k, 'bs ... (n_head d_head) -> bs ... n_head d_head', n_head=n_head)  # (bs ... lq_target n_head d_head)

        def pool(x):
            return einsum(x, "bs ... l_target n_head d_head -> bs l_target n_head d_head")

        q = self.q_norm(pool(q))  # (bs lq_target n_head d_head)
        k = self.k_norm(pool(k))  # (bs lk_target n_head d_head)

        #################### positional_encoding ##################
        if self.positional_encodings[dim_target] is not None:
            positional_encoding = self.positional_encodings[dim_target]

            # if self.token_positions == None:
            #     bs, lq_target, *_ = q.shape
            #     self.token_positions = torch.empty(
            #         size=(bs, n_head, lq_target), device=self.factory_kwargs["device"]
            #     )
            #     self.token_positions[:] = torch.arange(
            #         lq_target, device=self.factory_kwargs["device"]
            #     )
            
            # q = positional_encoding(x=q, token_positions=self.token_positions)
            # k = positional_encoding(x=k, token_positions=self.token_positions)           
        ############################################
        
        att = einsum(q, k, "bs lq_target n_head d_head, bs lk_target n_head d_head -> bs n_head lq_target lk_target") / math.sqrt(
            q.shape[-1]
        )

        #################### MASK ##################
        if mask is not None:
            # mask.shape = (bs, nh, lq_target, lk_target)
            # bs, *_ = q.shape
            # mask = torch.triu(
            #     torch.ones(
            #         (bs, n_head, lq_target, lk_target,
            #         dtype=torch.bool,
            #         device=self.factory_kwargs["device"],
            #     ),
            #     diagonal=1,
            # )
            # Boolean_mask should have shape "batch_size ... l1 l2"
            att = att.masked_fill(mask, float("-inf"))
        ############################################
        #att.shape = (bs n_head lq_target lk_target)
        logits = F.softmax(att, dim=-2)
        S = self.att_dropout(logits).float()
        h = einsum(S, v, "bs lq_target lk_target nh, bs ... l1 nh d -> bs ... lk_target nh d_head")
        return h.transpose(dim_target, -3), att
