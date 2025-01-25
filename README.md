<img src="./multistream.png" width="600px"></img>

Figure 1: Multi-Stream Transformer Encoder. a. Transformer encoder is a sequence of layers, we use this architecture as a baseline. We differentiate input transformer layer $L_{in}$, streams $S_i$ and output transformer layer Lout in the encoder. Baseline ”linear stream” model has only one stream $S_1$. Skip connection is created if output of the input stream $Z_{in}$ is added to the output Z1 of the linear stream. b. Multi-Stream encoder with two streams $S_1$ and $S_2$. Skip connection is shown as a dashed line.
## Multistream Transformers

Implementation of <a href="https://arxiv.org/abs/2107.10342">Multistream Transformers</a> in Pytorch.

This repository deviates slightly from the paper, where instead of using the skip connection across all streams, it uses attention pooling across all tokens in the same position. This has produced the best results in my experiments with number of streams greater than 2.

## Install

```
$ pip install multistream-transformers
```

## Usage

```python
import torch
from multistream_transformers import MultistreamTransformer

model = MultistreamTransformer(
    num_tokens = 256,         # number of tokens
    dim = 512,                # dimension
    depth = 4,                # depth
    causal = True,            # autoregressive or not
    max_seq_len = 1024,       # maximum sequence length
    num_streams = 2           # number of streams - 1 would make it a regular transformer
)

x = torch.randint(0, 256, (2, 1024))
mask = torch.ones((2, 1024)).bool()

logits = model(x, mask = mask) # (2, 1024, 256)
```

## Citations

```bibtex
@misc{burtsev2021multistream,
    title   = {Multi-Stream Transformers}, 
    author  = {Mikhail Burtsev and Anna Rumshisky},
    year    = {2021},
    eprint  = {2107.10342},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
