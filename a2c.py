import torch
import gym
import pickle as pkl

from basic_model import Actor, Critic
from buffer import Memory
from logger import StatsLogger


ENV_NAME = "Pendulum-v0"
ITERATIONS = 2000
GAMMA = 0.95
A_LR = 3e-3
C_LR = 3e-4
BATCH_SIZE = 1000
STATS_FREQ = 20
REWARD_DONE = 190.0
NUM_TARGET_UPDATES = 10
NUM_CRITIC_UPDATES = 10
NORMALIZE_ADV = True
FILENAME = 'a2c_logs_pendulum'


def train_a2c(
    env_name=ENV_NAME,
    iterations=ITERATIONS,
    gamma=GAMMA,
    a_lr=A_LR,
    c_lr=C_LR,
    stats_freq=STATS_FREQ,
    batch_size=BATCH_SIZE,
    reward_done=REWARD_DONE,
    num_target_updates=NUM_TARGET_UPDATES,
    num_critic_updates=NUM_CRITIC_UPDATES,
    normalize_adv=NORMALIZE_ADV,
    filename=FILENAME
):
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    actor = Actor(ob_dim, ac_dim, discrete)
    critic = Critic(ob_dim)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=a_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=c_lr)
    logger = StatsLogger()
    stats = []

    for i in range(iterations):
        buffer = Memory()
        collect_batch(env, actor, buffer, batch_size)
        update_critic(
            critic,
            critic_optimizer,
            buffer,
            gamma=gamma,
            num_target_updates=num_target_updates,
            num_critic_updates=num_critic_updates
        )

        advantages = calc_advantages(critic, buffer, gamma=gamma)
        update_actor(
            actor,
            actor_optimizer,
            advantages,
            buffer,
            normalize_adv=normalize_adv
        )

        running_reward = logger.calc_running_reward(buffer)

        if not i % stats_freq:
            logger.print_running_reward(i)
            stats.append([i, logger.running_reward])

        if running_reward >= reward_done:
            logger.task_done(i)
            break

        with open(filename + '.pkl', 'wb') as f:
            pkl.dump(stats, f)

    torch.save(actor.state_dict(), filename + '_model.pt')
    torch.save(critic.state_dict(), filename + '_critic_model.pt')

def collect_batch(
        env: gym.Env,
        actor: torch.nn.Module,
        buffer: Memory,
        batch_size: int
    ):

    while len(buffer) < batch_size:
        buffer.new_rollout()
        obs = env.reset()
        done = False
        prev_idx = buffer.add_obs(obs)

        while not done:
            obs = torch.unsqueeze(torch.FloatTensor(obs), dim=0)
            action, action_logprobs = actor.act(obs)
            action = action.numpy()[0]
            obs, rew, done, _ = env.step(action)
            next_idx = buffer.add_obs(obs)
            buffer.add_timestep(
                prev_idx,
                next_idx,
                action,
                action_logprobs,
                rew,
                done
            )
            prev_idx = next_idx

def update_critic(
        critic: torch.nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        buffer: Memory,
        gamma: float,
        num_target_updates: int,
        num_critic_updates: int
    ):

    obs = torch.tensor(buffer.obs, dtype=torch.float32)
    next_obs = torch.tensor(buffer.next_obs, dtype=torch.float32)
    reward = torch.tensor(buffer.rewards, dtype=torch.float32)
    done = torch.tensor(buffer.done, dtype=torch.float32)

    for _ in range(num_target_updates):
        next_state_value = critic(next_obs)
        next_state_value = next_state_value.detach()
        q_val = reward + gamma * (1 - done) * next_state_value.squeeze()

        for _ in range(num_critic_updates):
            state_value = critic(obs)
            advantage = q_val - state_value.squeeze()

            critic_loss = 0.5 * advantage.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

def calc_advantages(
        critic: torch.nn.Module,
        buffer: Memory,
        gamma: float,
    ) -> torch.Tensor:

    obs = torch.tensor(buffer.obs, dtype=torch.float32)
    next_obs = torch.tensor(buffer.next_obs, dtype=torch.float32)
    reward = torch.tensor(buffer.rewards, dtype=torch.float32)
    done = torch.tensor(buffer.done, dtype=torch.float32)

    with torch.no_grad(): # WARNING: should be changed in case of addition of dropout or other mode-dependent layers
        next_state_value = critic(next_obs)
        state_value = critic(obs)
        q_val = reward + gamma * (1 - done) * next_state_value.squeeze()
        advantages = q_val - state_value.squeeze()

    return advantages

def update_actor(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    advantages: torch.Tensor,
    buffer: Memory,
    normalize_adv: bool
    ):

    if normalize_adv:
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

    action_logprobs = buffer.action_logprobs
    action_logprobs = torch.cat(action_logprobs)

    actor_loss = (-action_logprobs * advantages).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


if __name__ == "__main__":
    train_a2c()
