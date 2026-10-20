# Matrix Normal Optimizers

PyTorch implementations of `Centered Adam`, `MN`, `MND`, and `MND-Muon` from
[*A Statistical Viewpoint on Adam, Shampoo and Muon*](https://arxiv.org/abs/2610.24839)
by Tetiana Parshakova, Robert M. Gower, and Paulo Orenstein (2026).

## Installation

Requires Python 3.10 or later and PyTorch 2.2 or later.

```bash
pip install -e .
```

## Usage

```python
import torch
from mnopt import CenteredAdam, MN, MND, MNDMuon

model = torch.nn.Linear(8, 4)
optimizer = MNDMuon(model.parameters(), lr=0.03)

x, y = torch.randn(32, 8), torch.randn(32, 4)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
optimizer.step()
```

Move the model to its target device before constructing the optimizer.
`MN`, `MND`, and `MND-Muon` apply matrix updates to two-dimensional parameters;
other parameters use `AdamW`. Set `matrix=False` in a parameter group to use
`AdamW` for embedding or output matrices. Centered Adam uses its own update
for every parameter.

Parameter settings are in [`configs/`](configs). Set `compile=True` to
compile the update kernel.

## Example

```bash
pip install -e '.[plot]'
python examples/regression.py --optimizer all --steps 100 
```

The regression example compares all four methods and three `MND-Muon` learning
rates. It displays loss curves without saving files.

## Citation

```bibtex
@misc{parshakova2026statistical,
  title  = {A Statistical Viewpoint on Adam, Shampoo and Muon},
  author = {Parshakova, Tetiana and Gower, Robert M. and Orenstein, Paulo},
  year   = {2026},
  eprint = {2610.24839},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2610.24839}
}
```

## License

[Apache License 2.0](LICENSE).
