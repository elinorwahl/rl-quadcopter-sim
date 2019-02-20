# Reinforcement Learning - Quadcopter Simulation

Reinforcement learning has become an exciting field of study in artificial intelligence in recent years. One of the most important reasons for this change is the development of **deep deterministic policy gradients**, or DDPG, which allow an AI agent to learn how to adapt its behavior to a continuous action space. In this project, we use DDPG to teach a simulated quadcopter - a drone with four rotors - how to fly. We give this quadcopter a target in 3D space to reach, and an approximation of real-world physics to give it a realistic challenge.

The simple way to explain DDPG is this: DDPG combines a **stochastic policy**, or a policy which takes actions based on a distribution of probability (for instance, there may be a 38% probability of a quadcopter flying down, and a 52% probability of the quadcopter flying up), with a **deterministic policy**, or one that directly maps states to actions (if the quadcopter descends beneath a certain height, it should go up). The stochastic policy guides the exploration of the agent, and the deterministic policy determines the targets the agent should aim for.

This project uses an **actor-critic model** to implement DDPG. In this case, there are two artificial neural networks, the ‘actor' and the ‘critic'. The ‘actor' uses its current state as input to take an action chosen from a continuous action space, and the ‘critic’ takes the actor’s output as its own input and produces an estimated Q-value, measured by the current state and the action taken by the actor, to determine how successful the action has been. A deterministic policy gradient uses the critic’s output to update the weights of the actor network, and the critic network is updated with the gradients from the temporal-difference error signal at each step. In this way, the actor, using a stochastic policy, and the critic, using a deterministic policy, learn from each other, and together they enable an AI agent to take better and more successful actions over time.

For me, one of the most impressive things about deep deterministic policy gradients and the actor-critic model is how little processing power they require to do what can be very complex tasks. This efficiency has enormous potential for the future of artificial intelligence, and for the future of energy use as well.

### Usage

Clone the Github repository and navigate to the downloaded folder:

```
git clone https://github.com/elinorwahl/rl-quadcopter-sim.git
cd rl-quadcopter-sim
```

Use [Anaconda](https://www.anaconda.com) to create and activate an environment:

```
conda create -n quadcopter python=3.6 matplotlib numpy pandas
source activate quadcopter
```

Create an [iPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the environment:

```
python -m ipykernel install —user —name quadcopter —display_name
```

And open the notebook:
```
jupyter notebook Quadcopter_project.ipynb
```

This project consists of:
- The `Quadcopter_Project.ipynb` notebook
- `policy_search.py` in the `agents` folder, a simplistic policy search agent
- The more complex `agent.py` in the `agents` folder, an Actor-Critic DDPG agent
- `physics_sim.py`, the physics simulator
- `quadcopter_plot.py`, which enables 3D plotting of the of the agent’s movements in space
- `task.py`, which defines the agent’s goal, action state spaces, reward function, and steps

### References

This is adapted from the quadcopter project in Udacity’s Deep Learning course, which can be found [here](https://github.com/udacity/RL-Quadcopter-2).

Much of the structure of the actor-critic model was obtained from the paper [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, _et al_.
