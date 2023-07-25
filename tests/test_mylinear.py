import unittest
import torch
import torchinfo
import cookies_models as cm


class TestMyLinear(unittest.TestCase):
    def test_(self):
        model = cm.MyLinear(4, 3)

        # test .summery()
        summary = model.summary()
        self.assertIs(type(summary), torchinfo.model_statistics.ModelStatistics)
        summary = model.summary((2, 4))
        self.assertIs(type(summary), torchinfo.model_statistics.ModelStatistics)

        # test .forward()
        input_ = torch.tensor([
            [1., 1., 1., 1.],
            [2., 2. ,2., 2.],
        ]).to('cuda')
        output_ = model(input_)
        self.assertEqual(tuple(output_.size()), (2, 3))  # test the output size
