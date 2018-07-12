#!/usr/bin/env python3
# -*- coding: utf-8
import torch
import numpy as np
from warp_loss import warp_loss, num_tries_gt_zero
import pytest


cpu_device = torch.device('cpu')
max_value = num_labels = 10

## warp-loss tests
def ground_truth(pos_val, neg_val, num_attempts=1, num_labels=10):
    num_labels -= 1
    loss_weight = np.log(np.floor(num_labels / float(num_attempts)))

    return loss_weight * ((1-pos_val) + neg_val)

comp_pos = torch.FloatTensor([[0.1], [0.9], [1]])
comp_neg = torch.FloatTensor([[-1, 0.3, 0.5], [-1, -1, 0.3], [0.3, 0.5, -1]])
comp_scores = (1 + comp_neg) - comp_pos

class TestWARPLoss:
    def test_num_tries(self):
        simple = torch.FloatTensor([[0.5, -0.5], [-0.5, 0.5]])
        res = num_tries_gt_zero(simple, 2, 2, max_value, cpu_device)
        ans = torch.LongTensor([1, 2])
        for i, v in enumerate(res.long()):
            assert v == ans[i]

        res = num_tries_gt_zero(comp_scores, 3, 3, max_value, cpu_device)
        ans = torch.LongTensor([2, 3, 1])
        for i, v in enumerate(res.long()):
            assert v == ans[i]


    def test_ground_truth(self):
        # these variables should always trigger on first index
        pos = torch.rand(2, 1).to(cpu_device)
        neg = torch.rand(2, 3).to(cpu_device)
        res = warp_loss(pos.view(-1, 1), neg, num_labels, cpu_device)

        assert res == ground_truth(pos[0], neg[0][0], num_labels=num_labels) + ground_truth(pos[1], neg[1][0], num_labels=num_labels)

        res = warp_loss(comp_pos.view(-1, 1), comp_neg, num_labels, cpu_device)

        gt = np.sum(np.array([ground_truth(comp_pos[i], comp_neg[i][idx-1], num_attempts=idx, num_labels=num_labels)
                              for i, idx in enumerate(num_tries_gt_zero(comp_scores, 3, 3, max_value, cpu_device))]))

        assert res.data.numpy() == pytest.approx(gt, 0.01)

    def test_no_offending_scores(self):
        pos = torch.FloatTensor([1, 1])
        neg = torch.FloatTensor([[-1, -1, -1],[-1, -1, -1]])
        res = warp_loss(pos.view(-1, 1), neg, num_labels, cpu_device)
        
        assert res == 0
