# TRANSFORMER MODEL
<b>Reference</b>: <i>Attention is All You Need</i>, Google Brain Team

## I. Model Architecture
<img src="./assets/Transformer.png"/>

## II. Details
### 1. Attention Mechanism <br/>
<b>Code: </b> <a href="https://github.com/Alan-404/Transformer-Model/blob/master/model/utils/attention.py">Attention Code</a> <br/>
Attention is the way that all items in a context can consider together and get attention weights for detecting their relations through multiple mechanism between query and key; And finally consider attention between this one and value.
Hyper - parameters of this layer is:
- heads: Number of heads used for Multi - Head Attention (heads = 1 is Self Attention). <br/>
- d_model: Number of embedding dimensions of word embedding.<br/>
<img src="./assets/attention.png"/>
Inputs of layer for forward propagation is:
- q: Play role query of context. <br/>
- k: Play role key of context.<br/>
- v: Play role value of context.<br/>
- mask: Hide the padding value for model.<br/>
