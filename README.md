<!--

#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: index.ipynb
# command to build the docs after a change: nbdev_build_docs

-->

# WARP-Pytorch

> An implementation of WARP loss which uses matrixes and stays on the GPU in PyTorch.


This means instead of using a for-loop to find the first offending negative sample that ranks above our positive,
we compute all of them at once. Only later do we find which sample is the first offender, and compute the loss with
respect to this sample.

The advantage is that it can use the speedups that comes with GPU-usage. 

## When is WARP loss advantageous?
If you're ranking items or making models for recommendations, it's often advantageous to let your loss function directly
optimize for this case. WARP loss looks at 1 explicit positive up against the implicit negative items that a user never sampled,
and allows us to adjust weights of the network accordingly.


## Install

`pip install warp_loss`

## How to use

The loss function requires scores for both positive examples, and negative examples to be supplied, such as in the example below.
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```
from torch import nn
import torch

class OurModel(nn.Module):
    def __init__(self, num_labels, emb_dim=10):
        super(OurModel, self).__init__()
        self.emb = nn.Embedding(num_labels, emb_dim)
        self.user_embs = nn.Embedding(1, emb_dim)

    def forward(self, pos, neg):
        batch_size = neg.size(0)
        one_user_vector = self.user_embs(torch.zeros(1).long())
        repeated_user_vector = one_user_vector.repeat((batch_size, 1)).view(batch_size, -1, 1)
        pos_res = torch.bmm(self.emb(pos), repeated_user_vector).squeeze(2)
        neg_res = torch.bmm(self.emb(neg), repeated_user_vector).squeeze(2)

        return pos_res, neg_res
        
num_labels = 100
model = OurModel(num_labels)
```

</div>

</div>
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```
pos_labels = torch.randint(high=num_labels, size=(3,1)) # our five labels
neg_labels = torch.randint(high=num_labels, size=(3,2)) # a few random negatives per positive

pos_res, neg_res = model(pos_labels, neg_labels)
print('Positive Labels:', pos_labels)
print('Negative Labels:', neg_labels)
print('Model positive scores:', pos_res)
print('Model negative scores:', neg_res)
loss = warp_loss(pos_res, neg_res, num_labels=num_labels, device=torch.device('cpu'))
print('Loss:', loss)
loss.backward()
```

</div>
<div class="output_area" markdown="1">

    Positive Labels: tensor([[21],
            [71],
            [26]])
    Negative Labels: tensor([[47, 10],
            [56, 78],
            [44, 55]])
    Model positive scores: tensor([[-4.9562],
            [-1.6886],
            [ 3.3984]], grad_fn=<SqueezeBackward1>)
    Model negative scores: tensor([[ 1.0491,  4.9357],
            [-2.1289,  0.4496],
            [ 3.4541,  0.0931]], grad_fn=<SqueezeBackward1>)
    Loss: tensor(39.6134, grad_fn=<SumBackward0>)


</div>

</div>
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```
print('We can also see that the gradient is only active for 2x the number of positive labels:', (model.emb.weight.grad.sum(1) != 0).sum().item())
print('Meaning we correctly discard the gradients for all other than the offending negative label.')
```

</div>
<div class="output_area" markdown="1">

    We can also see that the gradient is only active for 2x the number of positive labels: 6
    Meaning we correctly discard the gradients for all other than the offending negative label.


</div>

</div>

## Assumptions
The loss function assumes you have already sampled your negatives randomly.

As an example this could be done in your dataloader:

1. Assume we have a total dataset of 100 items
2. Select a positive sample with index 8
2. Your negatives should be a random selection from 0-100 excluding 8.

Ex input to loss function: model scores for pos: [8] neg: [88, 3, 99, 7]

Should work on all pytorch-versions from 0.4 and up

### References
* [WSABIE: Scaling Up To Large Vocabulary Image Annotation](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf)
* [Intro to WARP loss - Automatic differentiation and PyTorch](https://medium.com/@gabrieltseng/intro-to-warp-loss-automatic-differentiation-and-pytorch-b6aa5083187a)
* [LightFM](https://github.com/lyst/lightfm) as a reference implementaiton
