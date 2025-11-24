# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch


"""
Todo:
 none
"""



def create_transition_dtype(state_shape):
    """Create transition dtype based on state shape"""
    return np.dtype([
        ('timestep', np.int32), 
        ('state', np.float32, state_shape),
        ('action', np.int32), 
        ('reward', np.float32), 
        ('nonterminal', np.bool_)
    ])


def create_blank_transition(state_shape):
    """Create blank transition based on state shape"""
    return (0, np.zeros(state_shape, dtype=np.float32), 0, 0.0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size, state_shape):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        
        # Create dtype and blank transition dynamically based on state_shape
        self.transition_dtype = create_transition_dtype(state_shape)
        self.blank_trans = create_blank_transition(state_shape)
        
        self.data = np.array([self.blank_trans] * size, dtype=self.transition_dtype)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]  # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args, capacity, state_shape):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.state_shape = state_shape  # Store state shape
        
        # Recurrent network support
        self.use_recurrent = getattr(args, 'use_recurrent', False)
        self.recurrent_seq_len = getattr(args, 'recurrent_sequence_length', 8) if self.use_recurrent else 1
        
        self.n_step_scaling = torch.tensor(
            [self.discount ** i for i in range(self.n)], 
            dtype=torch.float32, 
            device=self.device
        )  # Discount-scaling vector for n-step returns
        
        # Pass state_shape to SegmentTree
        self.transitions = SegmentTree(capacity, state_shape)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        # Ensure state is numpy float32 array BEFORE converting to torch
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        elif state.dtype != np.float32:
            state = state.astype(np.float32)
        
        # Now convert to torch tensor
        state = torch.as_tensor(state, dtype=torch.float32, device='cpu')
        
        # Store new transition with maximum priority
        self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        if self.use_recurrent:
            # For recurrent networks, fetch longer sequences
            transition_idxs = np.arange(-self.recurrent_seq_len + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        else:
            # Standard behavior
            transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
            
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        
        if self.use_recurrent:
            # For recurrent: mask out transitions before episode start
            for t in range(self.recurrent_seq_len - 2, -1, -1):
                blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])
            for t in range(self.recurrent_seq_len, self.recurrent_seq_len + self.n):
                blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])
        else:
            # Standard behavior
            for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
                blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])
            for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
                blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])
                
        transitions[blank_mask] = self.transitions.blank_trans
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        
        # Determine the minimum required history
        min_history = self.recurrent_seq_len if self.use_recurrent else self.history
        
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= min_history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        
        # Retrieve all required transition data
        transitions = self._get_transitions(idxs)
        
        if self.use_recurrent:
            # Return sequences for recurrent networks
            return self._prepare_recurrent_batch(transitions, batch_size, probs, idxs, tree_idxs)
        else:
            # Standard non-recurrent batch preparation
            return self._prepare_standard_batch(transitions, batch_size, probs, idxs, tree_idxs)
    
    def _prepare_standard_batch(self, transitions, batch_size, probs, idxs, tree_idxs):
        """Prepare standard batch for non-recurrent networks"""
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(np.copy(all_states[:, :self.history]), device=self.device, dtype=torch.float32)
        next_states = torch.tensor(np.copy(all_states[:, self.n:self.n + self.history]), device=self.device, dtype=torch.float32)
        
        # Discrete actions to be used as index
        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
        
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals
    
    def _prepare_recurrent_batch(self, transitions, batch_size, probs, idxs, tree_idxs):
        """Prepare batch with sequences for recurrent networks"""
        # Extract state and action sequences
        # Sequences: from -(recurrent_seq_len-1) to 0 (current)
        state_sequences = transitions['state'][:, :self.recurrent_seq_len]  # (batch, seq_len, *state_shape)
        action_sequences = transitions['action'][:, :self.recurrent_seq_len]  # (batch, seq_len)
        
        # Next state sequences: for n-step ahead
        next_state_sequences = transitions['state'][:, self.n:self.n + self.recurrent_seq_len]
        next_action_sequences = transitions['action'][:, self.n:self.n + self.recurrent_seq_len]
        
        # Current action (the one at index recurrent_seq_len - 1, which is the "current" timestep)
        current_actions = torch.tensor(np.copy(transitions['action'][:, self.recurrent_seq_len - 1]), 
                                       dtype=torch.int64, device=self.device)
        
        # Calculate n-step returns
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.recurrent_seq_len - 1:-1]), 
                              dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        
        # Nonterminals
        nonterminals = torch.tensor(
            np.expand_dims(transitions['nonterminal'][:, self.recurrent_seq_len + self.n - 1], axis=1),
            dtype=torch.float32, device=self.device
        )
        
        # Create masks for valid timesteps (1 where valid, 0 where padded/blank)
        timesteps = transitions['timestep'][:, :self.recurrent_seq_len]
        # A timestep is invalid if it's 0 (episode start) AND it's not the very first step in our sequence
        # We need to create a mask that marks invalid positions
        masks = torch.ones((batch_size, self.recurrent_seq_len), dtype=torch.float32, device=self.device)
        next_masks = torch.ones((batch_size, self.recurrent_seq_len), dtype=torch.float32, device=self.device)
        
        # Convert sequences to tensors
        states_seq = torch.tensor(np.copy(state_sequences), device=self.device, dtype=torch.float32)
        actions_seq = torch.tensor(np.copy(action_sequences), device=self.device, dtype=torch.int64)
        next_states_seq = torch.tensor(np.copy(next_state_sequences), device=self.device, dtype=torch.float32)
        next_actions_seq = torch.tensor(np.copy(next_action_sequences), device=self.device, dtype=torch.int64)
        
        return (probs, idxs, tree_idxs, states_seq, actions_seq, masks, 
                current_actions, R, next_states_seq, next_actions_seq, next_masks, nonterminals)

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities
        samples = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
        
        if self.use_recurrent:
            # Unpack recurrent batch
            (probs, idxs, tree_idxs, states_seq, actions_seq, masks,
             actions, returns, next_states_seq, next_actions_seq, next_masks, nonterminals) = samples
        else:
            # Unpack standard batch
            probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = samples
        
        # Calculate importance-sampling weights
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        
        if self.use_recurrent:
            return (tree_idxs, states_seq, actions_seq, masks, actions, returns, 
                   next_states_seq, next_actions_seq, next_masks, nonterminals, weights)
        else:
            return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])  # If future frame has timestep 0
        transitions[blank_mask] = self.transitions.blank_trans
        state = torch.tensor(np.copy(transitions['state']), dtype=torch.float32, device=self.device)  # Agent will turn into batch
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility