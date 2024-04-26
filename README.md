
# RL Agent for Realistic Home Navigation and Interaction using AI2-THOR

Python library for implementing and experimenting with Continuously Updatable Policy Selection (CUPS) in Reinforcement Learning (RL) environments. CUPS-RL provides a framework for training and evaluating RL agents using various algorithms, including A3C, PPO, DDPG, and more. It also includes implementations of classic RL environments, as well as tools for visualizing and analyzing agent performance.

## Features

- Implementation of Continuously Updatable Policy Selection (CUPS) in RL environments.
- Support for various RL algorithms, including A3C, PPO, DDPG, Rainbow DQN, and more.
- Flexible and extensible architecture for defining custom RL environments and algorithms.
- Integration with popular RL libraries such as PyTorch and TensorFlow.
- Visualization tools for monitoring agent performance and training progress.

## Installation

To install, simply clone this repository and install the required dependencies using pip:

```bash
git clone https://github.com/tbharatha/AI2THOR_RL_PROJECT.git
pip install -r requirements.txt
```

## Usage

### Training an Agent in A3C 

To train an agent using a specific algorithm, you can use the provided training scripts in the `scripts` directory. For example, to train an A3C agent on the CartPole environment, you can run:

```bash
python algorithms/a3c/train.py
```

### Evaluating an Agent

You can evaluate a trained agent using the provided evaluation scripts. For example, to evaluate a trained A3C agent on the CartPole environment, you can run:

```bash
python algorithms/a3c/main.py
```
## Video Demonstration

[Watch the video](https://drive.google.com/file/d/1I9DpTnDmP4SARtjN-hKKHEv92KdTxiq-/view?usp=share_link)


### Similarly you can run Rainbow DQN Algotrithm.

Feel free to customize this template to better suit the specific features and goals of your project.
