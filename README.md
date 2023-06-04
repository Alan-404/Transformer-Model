# TRANSFORMER MODEL
<b>Reference</b>: <i>Attention is All You Need</i>, Google Brain Team

I. Model Architecture
<img src="./assets/Transformer.png"/>

II. Details
1. Attention Mechanism <br/>
<b>Code: </b> <a href="https://github.com/Alan-404/Transformer-Model/blob/master/model/utils/attention.py">Attention Code</a> <br/>
Attention is tha way that all items in a context can consider together and get attention weights for detecting their relations through matmul mechanism.
<img src="./assets/attention.png"/>
Input of layer is:
- q: Play role query of context.
- k: Play role key of context.
- v: Play role value of context.
- mask: Hide the padding value for model.
