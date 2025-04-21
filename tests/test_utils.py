import torch
from myosam.training.utils import (
    sample_initial_points,
    sample_points_from_error_region,
)


class TestAlgorithm:
    gt_masks = torch.randint(0, 2, (5, 1, 1024, 1024))
    pred_masks = torch.randint(0, 2, (5, 1, 1024, 1024))

    def test_sample_initial_points(self):
        coords, labels = sample_initial_points(self.gt_masks)
        assert coords.shape == (5, 1, 2)
        assert labels.shape == (5, 1)
        assert torch.all(labels == 1)
        assert torch.all(coords >= 0)
        assert torch.all(coords <= 1024)
        for i, coord in enumerate(coords):
            coord = coord.to(torch.int32)
            assert self.gt_masks[i, 0, coord[0, 1], coord[0, 0]] == 1

    def test_sample_points_from_error_region(self):
        coords, labels = sample_points_from_error_region(
            self.gt_masks, self.pred_masks
        )
        assert coords.shape == (5, 1, 2)
        assert labels.shape == (5, 1)
        assert torch.all(coords >= 0)
        assert torch.all(coords <= 1024)
        for i, (coord, label) in enumerate(zip(coords, labels)):
            coord = coord.to(torch.int32)
            if label == 1:
                assert self.gt_masks[i, 0, coord[0, 1], coord[0, 0]] == 1
                assert self.pred_masks[i, 0, coord[0, 1], coord[0, 0]] == 0
            elif label == 0:
                assert self.gt_masks[i, 0, coord[0, 1], coord[0, 0]] == 0
                assert self.pred_masks[i, 0, coord[0, 1], coord[0, 0]] == 1
            else:
                assert False
