# Finetuning-Llama2-with-LoRA
[Reference](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html) of this project is here. 

This guide will teach you about [LoRA](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html#lora-recipe-label), a parameter-efficient finetuning technique, and show you how you can use torchtune to finetune a Llama2 model with LoRA. If you already know what LoRA is and want to get straight to running your own LoRA finetune in torchtune, you can jump to [LoRA finetuning recipe in torchtune](https://arxiv.org/abs/2106.09685).

### What is LoRA?
[LoRA](https://arxiv.org/abs/2106.09685) is an adapter-based method for parameter-efficient finetuning that adds trainable low-rank decomposition matrices to different layers of a neural network, then freezes the network’s remaining parameters. LoRA is most commonly applied to transformer models, in which case it is common to add the low-rank matrices to some of the linear projections in each transformer layer’s self-attention.

### How does LoRA work?
LoRA replaces weight update matrices with a low-rank approximation. In general, weight updates for an arbitrary nn.Linear(in_dim,out_dim) layer could have rank as high as min(in_dim,out_dim). LoRA (and other related papers such as [Aghajanyan](https://arxiv.org/abs/2012.13255) et al.) hypothesize that the [intrinsic dimension](https://en.wikipedia.org/wiki/Intrinsic_dimension) of these updates during LLM fine-tuning can in fact be much lower. To take advantage of this property, LoRA finetuning will freeze the original model, then add a trainable weight update from a low-rank projection. More explicitly, LoRA trains two matrices A and B. A projects the inputs down to a much smaller rank (often four or eight in practice), and B projects back up to the dimension output by the original linear layer.

The image below gives a simplified representation of a single weight update step from a full finetune (on the left) compared to a weight update step with LoRA (on the right). The LoRA matrices A and B serve as an approximation to the full rank weight update in blue.

![Finetuning with LoRA](https://raw.githubusercontent.com/sulaiman-shamasna/Finetuning-Llama2-with-LoRA/main/image/image.png)
