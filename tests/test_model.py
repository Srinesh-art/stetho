import torch
from rural_stetho.model import DualViewRespiratoryNet

def test_dual_view_model_output_shape():
    model = DualViewRespiratoryNet(n_classes=4)
    mel = torch.randn(2, 64, 313)
    desc = torch.randn(2, 36)
    output = model(mel, desc)
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
