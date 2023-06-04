# TRANSFORMER MODEL
<b>Reference</b>: <i>Attention is All You Need</i>, Google Brain Team

## I. Model Architecture
<img src="./assets/Transformer.png"/>
    <table>
    <caption>Hyper - parameters of model</caption>
        <tr>
            <th>Name</th>
            <th>Value</th>
            <th>Meaning</th>
        </tr>
        <tr>
            <td>encoder_token_size</td>
            <td>Depend on tokenization</td>
            <td>Number of tokens used for Encoder</td>
        </tr>
        <tr>
            <td>decoder_token_size</td>
            <td>Depend on tokenization</td>
            <td>Number of tokens used for Decoder</td>
        </tr>
        <tr>
            <td>n</td>
            <td>6</td>
            <td>Number of Encoder layers, so does Decoder layers</td>
        </tr>
        <tr>
            <td>heads</td>
            <td>8</td>
            <td>Number of heads for Multi - Head Attention</td>
        </tr>
        <tr>
            <td>d_model</td>
            <td>512</td>
            <td>Number of dimensions in word embedding</td>
        </tr>
        <tr>
            <td>d_ff</td>
            <td>2048</td>
            <td>Number of hidden dimensions in Position wise Feed Forward Networks</td>
        </tr>
        <tr>
            <td>dropout_rate</td>
            <td>0.1</td>
            <td>Probability of Dropout Layer</td>
        </tr>
        <tr>
            <td>eps</td>
            <td>0.1</td>
            <td>Coefficient Normalization of Layer Norm</td>
        </tr>
        <tr>
            <td>activation</td>
            <td>ReLU</td>
            <td>Activation function in Position wise Feed Forwards Networks</td>
    </tr>
</table>

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
