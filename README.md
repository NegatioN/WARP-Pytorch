# WARP loss for PyTorch

An implementation of WARP loss which uses matrixes and stays on the GPU in PyTorch.

This means instead of using a for-loop to find the first offending negative sample that ranks above our positive,
we compute all of them at once. Only later do we find which sample is the first offender, and compute the loss with
respect to this sample.

The advantage is that it can use the speedups that comes with GPU-usage. 

## Assumptions
The loss function assumes you have already sampled your negatives randomly.

As an example this could be done in your dataloader:

1. Assume we have a total dataset of 100 items
2. Select a positive sample with index 8
2. Your negatives should be a random selection from 0-100 excluding 8.

Ex input to loss function: model scores for pos: [8] neg: [88, 3, 99, 7]

Currently only tested on PyTorch v0.4
