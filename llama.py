from lora import LoRALinear
from torchtune.models.llama2 import llama2_7b, lora_llama2_7b

# Build Llama2 without any LoRA layers
base_model = llama2_7b()

# The default settings for lora_llama2_7b will match those for llama2_7b
# We just need to define which layers we want LoRA applied to.
# Within each self-attention, we can choose from ["q_proj", "k_proj", "v_proj", and "output_proj"].
# We can also set apply_lora_to_mlp=True or apply_lora_to_output=True to apply LoRA to other linear
# layers outside of the self-attention.
lora_model = lora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])

print(base_model.layers[0].attn)

""" OUTPUT:

CausalSelfAttention(
  (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (pos_embeddings): RotaryPositionalEmbeddings()
)
"""

print(lora_model.layers[0].attn)

""" OUTPUT:

CausalSelfAttention(
  (q_proj): LoRALinear(
    (dropout): Dropout(p=0.0, inplace=False)
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=4096, bias=False)
  )
  (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (v_proj): LoRALinear(
    (dropout): Dropout(p=0.0, inplace=False)
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=4096, bias=False)
  )
  (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (pos_embeddings): RotaryPositionalEmbeddings()
)
"""

# Assuming that base_model already has the pretrained Llama2 weights,
# this will directly load them into your LoRA model without any conversion necessary.
lora_model.load_state_dict(base_model.state_dict(), strict=False)


from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params

# Fetch all params from the model that are associated with LoRA.
lora_params = get_adapter_params(lora_model)

# Set requires_grad=True on lora_params, and requires_grad=False on all others.
set_trainable_params(lora_model, lora_params)

# Print the total number of parameters
total_params = sum([p.numel() for p in lora_model.parameters()])
trainable_params = sum([p.numel() for p in lora_model.parameters() if p.requires_grad])
print(
  f"""
  {total_params} total params,
  {trainable_params}" trainable params,
  {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
  """
)

""" OUTPUT:

6742609920 total params,
4194304" trainable params,
0.06% of all params are trainable.
"""