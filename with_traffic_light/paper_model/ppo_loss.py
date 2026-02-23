import torch
import torch.nn.functional as F

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0

    # Iterate backwards through the collected trajectory
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_val = values[step + 1]

        # Calculate Temporal Difference (TD) error
        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)

    # Normalize advantages for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def ppo_update(agent, optimizer, states, actions, old_log_probs, returns, advantages, clip_param=0.2, entropy_coef=0.01, value_coef=0.5, epochs=4):
    # Align advantage tensor shape with ratios
    advantages = advantages.unsqueeze(1)

    for _ in range(epochs):
        dist, state_values = agent(states)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(state_values.squeeze(), returns)

        total_loss = actor_loss + (value_coef * critic_loss) - (entropy_coef * entropy)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()