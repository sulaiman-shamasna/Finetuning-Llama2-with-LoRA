# Finetuning-Llama2-with-LoRA
[Reference](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html) of this project is here. 

This guide will teach you about [LoRA](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html#lora-recipe-label), a parameter-efficient finetuning technique, and show you how you can use torchtune to finetune a Llama2 model with LoRA. If you already know what LoRA is and want to get straight to running your own LoRA finetune in torchtune, you can jump to [LoRA finetuning recipe in torchtune](https://arxiv.org/abs/2106.09685).

### What is LoRA?
[LoRA](https://arxiv.org/abs/2106.09685) is an adapter-based method for parameter-efficient finetuning that adds trainable low-rank decomposition matrices to different layers of a neural network, then freezes the network’s remaining parameters. LoRA is most commonly applied to transformer models, in which case it is common to add the low-rank matrices to some of the linear projections in each transformer layer’s self-attention.

### How does LoRA work?
LoRA replaces weight update matrices with a low-rank approximation. In general, weight updates for an arbitrary nn.Linear(in_dim,out_dim) layer could have rank as high as min(in_dim,out_dim). LoRA (and other related papers such as [Aghajanyan](https://arxiv.org/abs/2012.13255) et al.) hypothesize that the [intrinsic dimension](https://en.wikipedia.org/wiki/Intrinsic_dimension) of these updates during LLM fine-tuning can in fact be much lower. To take advantage of this property, LoRA finetuning will freeze the original model, then add a trainable weight update from a low-rank projection. More explicitly, LoRA trains two matrices A and B. A projects the inputs down to a much smaller rank (often four or eight in practice), and B projects back up to the dimension output by the original linear layer.

The image below gives a simplified representation of a single weight update step from a full finetune (on the left) compared to a weight update step with LoRA (on the right). The LoRA matrices A and B serve as an approximation to the full rank weight update in blue.

![Finetuning with LoRA](https://raw.githubusercontent.com/sulaiman-shamasna/Finetuning-Llama2-with-LoRA/main/image/image.png)

Although LoRA introduces a few extra parameters in the model ```forward()```, only the A and B matrices are trainable. This means that with a rank r LoRA decomposition, the number of gradients we need to store reduces from ```in_dim*out_dim``` to ```r*(in_dim+out_dim)```. (Remember that in general r is much smaller than ```in_dim``` and ```out_dim```.)

For example, in the 7B Llama2’s self-attention, ```in_dim=out_dim=4096``` for the Q, K, and V projections. This means a LoRA decomposition of rank r=8 will reduce the number of trainable parameters for a given projection from 
 to 
, a reduction of over 99%.

### Implementation with PyTorch
Let’s take a look at a minimal implementation of LoRA in native PyTorch.
1. **Create a LoRA layer** (check [this](https://github.com/sulaiman-shamasna/Finetuning-Llama2-with-LoRA/blob/main/lora.py)) for the complete code, which looks something like:
```python
from torch import nn, Tensor

class LoRALinear(nn.Module):
  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    rank: int,
    alpha: float,
    dropout: float
  ):
  ...
```
There are some other details around initialization which we omit here, but if you’d like to know more you can see our implementation in [LoRALinear](https://pytorch.org/torchtune/stable/generated/torchtune.modules.peft.LoRALinear.html#torchtune.modules.peft.LoRALinear). Now that we understand what LoRA is doing, let’s look at how we can apply it to our favorite models.

2. **Applying LoRA to Llama2 models**. With torchtune, we can easily apply LoRA to Llama2 with a variety of different configurations. Let’s take a look at how to construct Llama2 models in torchtune with and without LoRA. Complete code can be found [here](https://github.com/sulaiman-shamasna/Finetuning-Llama2-with-LoRA/blob/main/llama.py).
```python
from torchtune.models.llama2 import llama2_7b, lora_llama2_7b

# Build Llama2 without any LoRA layers
base_model = llama2_7b()
lora_model = lora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])
```

**Note**: Calling ```lora_llama_2_7b``` alone will not handle the definition of which parameters are trainable. See [here/ bellow](https://github.com/sulaiman-shamasna/Finetuning-Llama2-with-LoRA/blob/main/llama.py) for how to do this.

```python
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
```
Notice that our LoRA model’s layer contains additional weights in the Q and V projections, as expected. Additionally, inspecting the type of ```lora_model``` and ```base_model```, would show that they are both instances of the same [TransformerDecoder](https://pytorch.org/torchtune/stable/generated/torchtune.modules.TransformerDecoder.html#torchtune.modules.TransformerDecoder). 