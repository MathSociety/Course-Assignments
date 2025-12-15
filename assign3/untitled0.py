import numpy as np
import random

def TripleQLearning(env, beta, Nepisodes, alpha):
    epsilon = 0.1
    n_actions = 2
            
    q_matrix = np.zeros((3, 21, 21, n_actions, 10, 2))  # Changed from q_array
        
    def select_action(s):
        if random.random() < epsilon:
            return random.randint(0, n_actions - 1)
        else:
            q_values_total = np.sum(q_matrix[:, s[0], s[1], :, s[3], s[2]], axis=0)
            return np.argmax(q_values_total)
            
    def apply_update(s, act, rew, s_next):
        index = random.randint(0, 2)
        others = {0, 1, 2} - {index}
            
        q_x, q_y = [q_matrix[u, s_next[0], s_next[1], act, s_next[3], s_next[2]] for u in others]
        avg_q_next = (q_x + q_y) / 2

        q_matrix[index, s[0], s[1], act, s[3], s[2]] += alpha * (
            rew + (beta) * avg_q_next - q_matrix[index, s[0], s[1], act, s[3], s[2]]
        )
        
    for episode_num in range(Nepisodes):
        s = env.reset()
        completed = False
                
        print(f"Episode: {episode_num + 1}")
        while not completed:
            act = select_action(s)
            s_next, rew, completed, _, _ = env.step(act)
            apply_update(s, act, rew, s_next)
            s = s_next
            env.render()
    return q_matrix

