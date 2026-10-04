import numpy as np
import torch

from clin2pro.data import Standardizer, canonical_protein_name
from clin2pro.losses import masked_huber, masked_pathway_profiles
from clin2pro.models import LowRankProteinTargetBridge


def test_canonical_protein_name():
    assert canonical_protein_name("ABL1;Tyrosine kinase") == "ABL1"


def test_standardizer_handles_missing_and_constant():
    values = np.asarray([[1.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    scaler = Standardizer.fit(values)
    transformed = scaler.transform(values)
    assert np.isfinite(transformed).all()
    assert scaler.scale[0] == 1.0


def test_masked_huber_ignores_missing_position():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    assert torch.isclose(masked_huber(prediction, target, mask), torch.tensor(0.5))


def test_pathway_profile_uses_sample_specific_denominator():
    values = torch.tensor([[2.0, 10.0]])
    mask = torch.tensor([[True, False]])
    membership = torch.tensor([[1.0], [1.0]])
    profile, pathway_mask = masked_pathway_profiles(values, mask, membership)
    assert torch.isclose(profile[0, 0], torch.tensor(2.0))
    assert pathway_mask[0, 0]


def test_low_rank_bridge_shape_and_factorization():
    bridge = LowRankProteinTargetBridge(5, 3, 2)
    x = torch.randn(4, 5)
    output = bridge(x)
    assert output.shape == (4, 3)
    assert torch.allclose(output, x.matmul(bridge.matrix()))
