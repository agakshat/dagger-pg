import sys
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import gym
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks import ActorNetwork
import pdb

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='LunarLander-v2',
                        help='environment to train on (default: LunarLander-v2)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed (default: 12345)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render environment')

    return parser.parse_args()


def main(args):
    args = parse_arguments()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    env = gym.make(args.env_name)
    os.environ['OMP_NUM_THREADS'] = '1'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    actor = ActorNetwork(env.observation_space.shape[0],env.action_space.n)
    if args.cuda:
        actor.cuda()
    optimizer = optim.Adam(actor.parameters(),lr=args.lr)
    
    eps = 0.5
    total_num_updates = 0
    for ep in range(args.num_episodes):
        done = False
        obs =  env.reset()
        if eps>0.05: # linearly decaying greedy parameter epsilon
            eps = -0.0009*ep + 0.5

        obsarr = []
        rewardarr = []
        donearr = []
        actionarr = []
        logprobarr = []

        while not done:
            obs_var = Variable(torch.from_numpy(obs).float())
            _,action_probs = actor.get_action(obs_var)
            if np.random.random()<eps:
                action = env.action_space.sample()
            else:
                _,action = torch.max(action_probs,-1)
                action = action.data[0]

            log_prob = action_probs.log()[action]
            next_obs,reward,done,info = env.step(action)
            if args.render:
                env.render()
            obsarr.append(obs)
            rewardarr.append(reward)
            donearr.append(done)
            actionarr.append(action)
            logprobarr.append(log_prob)
            obs = next_obs

        T = len(obsarr)
        G = [0]*T
        G[T-1] = rewardarr[T-1]
        for t in reversed(range(T-1)):
            G[t] = args.gamma*G[t+1] + rewardarr[t]
        Gtensor = Variable(torch.FloatTensor(G))
        logprobvar = torch.cat(logprobarr)
        loss = (0.01*Gtensor*logprobvar).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(actor.parameters(),3)
        optimizer.step()

        print("Episode: {} | Reward: {:.3f}| Epsilon: {:.3f}".format(ep,np.array(rewardarr).mean(),eps))



if __name__ == '__main__':
    main(sys.argv)
