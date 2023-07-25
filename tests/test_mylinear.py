import torch
# import torchinfo
import cookies_models as cm


class TestMyLinear:
    def test_(self):
        model = cm.MyLinear(4, 3)

        # test .summery()
        # summary = model.summary()
        # assert type(summary) is torchinfo.model_statistics.ModelStatistics
        # summary = model.summary((2, 4))
        # assert type(summary) is torchinfo.model_statistics.ModelStatistics

        # test .forward()
        input_ = torch.tensor([
            [1., 1., 1., 1.],
            [2., 2. ,2., 2.],
        ]).to('cuda')
        output_ = model(input_)
        assert tuple(output_.size()) == (2, 3)  # test the output size
