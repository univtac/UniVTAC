import torch


class DummyVQLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        zero = torch.zeros((), device=args[1].device if len(args) > 1 and hasattr(args[1], "device") else "cpu")
        return zero, {}
