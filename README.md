# Eik-HIQL
Official implementation of Physics-informed Value Learner for Scalable Offline Goal-Conditioned Reinforcement Learning (Eikonal-regularized Hierarchical Implicit Q-learning (Eik-HIQL)) accepted as a conference paper in NeurIPS 2025. 

### Installation

Our implementations require Python 3.9+ and additional dependencies, including `jax >= 0.4.26`.
To install these dependencies, run:

```shell
pip install -r requirements.txt
```

### Running Eik-HIQL

To train an agent, you can run the `main.py` script.
Training metrics, evaluation metrics, and videos are logged via `wandb` by default.
Here are some example commands (see [hyperparameters.sh](hyperparameters.sh) for the full list of commands):

```shell
# antmaze-large-navigate-v0 (Eik-HIQL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/pi_hiql.py 
```

### Acknowledgements

We want to thank the authors of the [OGBench](https://seohong.me/projects/ogbench/) benchmark for their amazing work.
Most of this code is based on their implementations. Please refer also to their [GitHub Repo](https://github.com/seohongpark/ogbench).

### Citations

If you found our paper or code useful, please cite

```
@article{giammarino2025physics,
  title={Physics-informed Value Learner for Offline Goal-Conditioned Reinforcement Learning},
  author={Giammarino, Vittorio and Ni, Ruiqi and Qureshi, Ahmed H},
  journal={arXiv preprint arXiv:2509.06782},
  year={2025}
}
```
