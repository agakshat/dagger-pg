import gym
import torch
import numpy as np
from networks import ActorNetwork
import argparse
from torch.autograd import Variable

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='LunarLander-v2',
                        help='environment to train on (default: LunarLander-v2)')
    parser.add_argument('--load-dir', default='.',
                        help='directory to model path (default: .)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render environment')
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50, help="Number of episodes to test.")
    return parser.parse_args()

def test(env,actor,render):
    rew_arr = []
    ep_len_arr = []
    for ep in range(100):
        ep_len = 0
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            ep_len += 1
            obs_var = Variable(torch.from_numpy(obs).float())
            action = actor.get_action(obs_var)
            #if np.random.random()<eps:
            #    action = env.action_space.sample()
            #else:
            #    _,action = torch.max(action_probs,-1)
            #    action = action.data[0]
            action = action.data[0]
            next_obs,reward,done,_ = env.step(action)
            if render:
                env.render()
            ep_reward += reward
            obs = next_obs
        ep_len_arr.append(ep_len)
        rew_arr.append(ep_reward)
        print("Reward: {:.3f}| Length: {:.3f}".format(ep_reward,ep_len))

    print("Reward Mean: {:.3f}, Std: {:.3f}| Length: {:.3f}".format(
            np.array(rew_arr).mean(),np.array(rew_arr).std(),
            np.array(ep_len_arr).mean()))
    return np.array(rew_arr).mean(),np.array(rew_arr).std(),np.array(ep_len_arr).mean()

def main():
  args = parse_arguments()
  env = gym.make(args.env_name)
  actor = ActorNetwork(env.observation_space.shape[0],env.action_space.n)
  actor.load_state_dict(torch.load(args.load_dir))
  test(env,actor,args.render)

if __name__=='__main__':
  main()