import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Union, Optional


class DiagonalLegPenaltyWrapper(gym.Wrapper):
    """
    A wrapper for the Ant environment that penalizes the use of diagonal legs.
    
    Based on Gymnasium documentation, the Ant has 4 legs with actions:
    - Leg 1: Front left (actions 2,3) - hip_1, angle_1
    - Leg 2: Front right (actions 4,5) - hip_2, angle_2  
    - Leg 3: Back left (actions 6,7) - hip_3, angle_3
    - Leg 4: Back right (actions 0,1) - hip_4, angle_4
    
    This wrapper can penalize either:
    - Diagonal 1: Legs 1 and 4 (front left + back right)
    - Diagonal 2: Legs 2 and 3 (front right + back left)
    """
    
    def __init__(self, 
                 env: gym.Env, 
                 diagonal_type: int = 1,
                 penalty_weight: float = -1.0):
        """
        Args:
            env: The Ant environment to wrap
            diagonal_type: Which diagonal to penalize (1 or 2)
            penalty_weight: Weight of the penalty (negative value)
        """
        super().__init__(env)
        
        if diagonal_type not in [1, 2]:
            raise ValueError("diagonal_type must be 1 or 2")
        
        self.diagonal_type = diagonal_type
        self.penalty_weight = penalty_weight
        
        # Define which actions correspond to which diagonal
        if diagonal_type == 1:
            # Penalize legs 1 and 4 (front left + back right)
            self.penalized_actions = [2, 3, 0, 1]  # hip_1, angle_1, hip_4, angle_4
        else:
            # Penalize legs 2 and 3 (front right + back left)  
            self.penalized_actions = [4, 5, 6, 7]  # hip_2, angle_2, hip_3, angle_3
        
        print(f"Penalizing diagonal {diagonal_type} (actions {self.penalized_actions})")
        print(f"Penalty weight: {penalty_weight}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment and apply diagonal leg penalty.
        
        Args:
            action: Action array of shape (8,) for the 8 joints
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get the original step result
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate diagonal leg penalty
        diagonal_penalty = self._calculate_diagonal_penalty(action)
        
        # Apply penalty to reward
        modified_reward = reward + diagonal_penalty
        
        # Store penalty information in info
        info['diagonal_penalty'] = diagonal_penalty
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        info['diagonal_type'] = self.diagonal_type
        
        return obs, modified_reward, terminated, truncated, info
    
    def _calculate_diagonal_penalty(self, action: np.ndarray) -> float:
        """
        Calculate penalty based on usage of penalized diagonal legs.
        
        Args:
            action: Action array of shape (8,)
            
        Returns:
            Penalty value (negative)
        """
        # Calculate the squared magnitude of actions for penalized joints
        penalized_squared = action[self.penalized_actions] ** 2
        
        # Simple penalty: sum of all penalized action squared values
        total_penalty = np.sum(penalized_squared)
        
        return self.penalty_weight * total_penalty
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        return obs, info


def create_ant_with_diagonal_penalty(env_name: str = "Ant-v4", 
                                   diagonal_type: int = 1,
                                   penalty_weight: float = -1.0) -> DiagonalLegPenaltyWrapper:
    """
    Convenience function to create an Ant environment with diagonal leg penalty.
    
    Args:
        env_name: Name of the Ant environment
        diagonal_type: Which diagonal to penalize (1 or 2)
        penalty_weight: Weight of the penalty (negative value)
        
    Returns:
        Wrapped Ant environment with diagonal leg penalty
    """
    import gymnasium as gym
    
    base_env = gym.make(env_name)
    return DiagonalLegPenaltyWrapper(
        base_env, 
        diagonal_type=diagonal_type,
        penalty_weight=penalty_weight
    ) 