

import torch
import numpy as np


def create_ppo_agent(
    obs_dim,
    action_space,
    action_space_type,
    n_spokes,
    hidden_dim,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    num_epochs=10,
    eps_clip=0.2,
    action_std_init=0.3,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    batch_size=64,
    max_grad_norm=0.5,
    device='cpu',
    use_recurrent=False,
    recurrent_type='lstm',
    recurrent_hidden_dim=128,
    recurrent_layers=1,
    recurrent_sequence_length=16,
    recurrent_dropout=0.0,
):

    
    print(f"\n[PPO Factory] Creating agent for action_space_type: {action_space_type}")
    
    if action_space_type == 'discrete':
        # Standard discrete PPO
        from Agents import PPOAgent
        
        action_dim = action_space.n
        print(f"[PPO Factory] Using standard PPOAgent with {action_dim} discrete actions")
        
        return PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            continuous_action_space=False,
            num_epochs=num_epochs,
            eps_clip=eps_clip,
            action_std_init=action_std_init,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
            use_recurrent=use_recurrent,
            recurrent_type=recurrent_type,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_sequence_length=recurrent_sequence_length,
            recurrent_dropout=recurrent_dropout,
        )
    
    elif action_space_type == 'hybrid' or action_space_type == 'continous':
        # Hybrid: discrete spoke selection + continuous delta
        from hybrid_actor_critic import HybridPPOAgent
        
        print(f"[PPO Factory] Using HybridPPOAgent with {n_spokes} spokes + continuous delta")
        
        return HybridPPOAgent(
            obs_dim=obs_dim,
            n_spokes=n_spokes,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            num_epochs=num_epochs,
            eps_clip=eps_clip,
            delta_std_init=action_std_init,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
            use_recurrent=use_recurrent,
            recurrent_type=recurrent_type,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_sequence_length=recurrent_sequence_length,
            recurrent_dropout=recurrent_dropout,
        )
    
    elif action_space_type == 'all_spokes':
        # Continuous: adjust all spokes at once
        from Agents import PPOAgent
        
        action_dim = action_space.shape[0]  # n_spokes
        print(f"[PPO Factory] Using standard PPOAgent with {action_dim} continuous actions (all spokes)")
        
        return PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            continuous_action_space=True,
            num_epochs=num_epochs,
            eps_clip=eps_clip,
            action_std_init=action_std_init,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
            use_recurrent=use_recurrent,
            recurrent_type=recurrent_type,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_sequence_length=recurrent_sequence_length,
            recurrent_dropout=recurrent_dropout,
        )
    
    else:
        raise ValueError(f"Unknown action_space_type: {action_space_type}. "
                        f"Must be one of: 'discrete', 'continuous', 'hybrid', 'all_spokes'")
