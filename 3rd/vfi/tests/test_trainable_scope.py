import unittest

import torch
from torch import nn

from Trainer import Model


class _TinyFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_bone = nn.Linear(2, 2)
        self.block = nn.ModuleList([nn.Linear(2, 2)])
        self.local_block = nn.ModuleList([
            nn.Linear(2, 2),
            nn.ModuleDict({
                'base': nn.Linear(2, 2),
                'content_upsampler': nn.Linear(2, 2),
            }),
        ])
        self.sparse_matching_feature_encoder = nn.Sequential(
            nn.Linear(2, 2))
        self.sparse_matcher = nn.Sequential(nn.Linear(2, 2))
        self.multi_hypothesis = nn.Sequential(nn.Linear(2, 2))
        self.unet = nn.Linear(2, 2)


class TrainableScopeTest(unittest.TestCase):
    def test_flow_heads_freezes_backbone_and_refiner(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)

        model.set_trainable_scope('flow_heads')

        states = {
            name: parameter.requires_grad
            for name, parameter in model.net.named_parameters()
        }
        self.assertTrue(states['block.0.weight'])
        self.assertTrue(states['local_block.0.weight'])
        self.assertTrue(states['local_block.1.base.weight'])
        self.assertTrue(states['local_block.1.content_upsampler.weight'])
        self.assertFalse(states['feature_bone.weight'])
        self.assertFalse(states['unet.weight'])

    def test_sparse_matcher_scope_only_enables_matching_branch(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)

        model.set_trainable_scope('sparse_matcher')

        states = {
            name: parameter.requires_grad
            for name, parameter in model.net.named_parameters()
        }
        self.assertTrue(states['sparse_matcher.0.weight'])
        self.assertFalse(states['local_block.0.weight'])
        self.assertFalse(states['block.0.weight'])
        self.assertFalse(states['feature_bone.weight'])
        self.assertFalse(states['unet.weight'])

    def test_all_scope_keeps_pretrained_matching_features_frozen(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)

        model.set_trainable_scope('all')

        states = {
            name: parameter.requires_grad
            for name, parameter in model.net.named_parameters()
        }
        self.assertTrue(states['feature_bone.weight'])
        self.assertTrue(states['sparse_matcher.0.weight'])
        self.assertFalse(
            states['sparse_matching_feature_encoder.0.weight'])

    def test_multi_hypothesis_scope_only_enables_candidate_branch(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)

        model.set_trainable_scope('multi_hypothesis')

        states = {
            name: parameter.requires_grad
            for name, parameter in model.net.named_parameters()
        }
        self.assertTrue(states['multi_hypothesis.0.weight'])
        self.assertFalse(states['sparse_matcher.0.weight'])
        self.assertFalse(states['local_block.0.weight'])
        self.assertFalse(states['block.0.weight'])
        self.assertFalse(states['feature_bone.weight'])
        self.assertFalse(states['unet.weight'])

    def test_caun_scope_only_enables_content_upsampler(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)

        model.set_trainable_scope('caun')

        states = {
            name: parameter.requires_grad
            for name, parameter in model.net.named_parameters()
        }
        self.assertTrue(states['local_block.1.content_upsampler.weight'])
        self.assertFalse(states['local_block.1.base.weight'])
        self.assertFalse(states['local_block.0.weight'])
        self.assertFalse(states['block.0.weight'])
        self.assertFalse(states['feature_bone.weight'])
        self.assertFalse(states['unet.weight'])

    def test_unknown_scope_is_rejected(self):
        model = object.__new__(Model)
        model.net = _TinyFlowNet()
        model.optimG = torch.optim.AdamW(model.net.parameters(), lr=1e-4)
        with self.assertRaises(ValueError):
            model.set_trainable_scope('flow_everything')


if __name__ == '__main__':
    unittest.main()
