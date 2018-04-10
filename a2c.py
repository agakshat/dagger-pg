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
from networks import ActorNetwork,CriticNetwork
import pdb
from tensorboardX import SummaryWriter

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='LunarLander-v2',
                        help='environment to train on (default: LunarLander-v2)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed (default: 12345)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--save-dir', default='./trained_models',
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
    parser.add_argument('--update-freq', type=int, default=1,
                        help='how frequently to update network (default: 1)')
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='continue training from another model')
    parser.add_argument('--load-dir', default='./trained_models/',
                        help='path to trained model file, if available')
    parser.add_argument('--nsteps', type=int, default=5,
                        help='a2c steps')
    parser.add_argument('--value-loss-coeff', type=float, default=0.5,
                        help='value loss coefficient')
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
    writer = SummaryWriter(log_dir=args.save_dir)
    actor = ActorNetwork(env.observation_space.shape[0],env.action_space.n)
    critic = CriticNetwork(env.observation_space.shape[0])
    if args.continue_training:
        try:
            actorState = torch.load(args.load_dir,map_location = lambda storage, loc: storage)
            actor.load_state_dict(actorState)
        except:
            assert False, "Unable to find a model to load"
    if args.cuda:
        actor.cuda()
        critic.cuda()
    actor_optimizer = optim.Adam(actor.parameters(),lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(),lr=args.lr)
    N = args.nsteps
    eps = 1.0
    obsarr = []
    rewardarr = []
    actionlossarr = []
    actionarr = []
    valuearr = []
    ep_len = 0
    for ep in range(args.num_episodes):
        done = False
        obs =  env.reset()
        
        while not done:
            ep_len += 1
            obs_var = Variable(torch.from_numpy(obs).float(),volatile=True)
            action = actor.get_action(obs_var)
            value = critic(obs_var)
            #if np.random.random()<eps:
            #    action = env.action_space.sample()
            #else:
            #    _,action = torch.max(action_probs,-1)
            #    action = action.data[0]
            action = action.data[0]
            next_obs,reward,done,_ = env.step(action)
            if args.render:
                env.render()
            obsarr.append(obs)
            actionarr.append(action)
            rewardarr.append(reward)
            valuearr.append(value)
            obs = next_obs

        T = len(obsarr)
        G = [0]*T

        batch_obs = Variable(torch.from_numpy(np.stack(obsarr)).float())
        batch_act = Variable(torch.from_numpy(np.array(actionarr)))
        logprobvar = actor.evaluate_actions(batch_obs,batch_act)
        valvar = critic(batch_obs)
        logprobvar = logprobvar.squeeze(1)
        valvar = valvar.squeeze(1)

        for t in reversed(range(T)):
            V = 0
            if t+N<T:
                V = valvar[t+N].data[0]
            G[t] = pow(args.gamma,N)*V
            u = min(N,T-t)
            for k in range(u):
                G[t] += pow(args.gamma,k)*rewardarr[t+k]

        Gtensor = Variable(torch.FloatTensor(G))
        adv = 0.01*Gtensor - valvar.detach()
        action_loss = -(adv*logprobvar).mean()
        value_loss = (0.01*Gtensor-valvar).pow(2).mean()
        #pdb.set_trace()
        #loss = action_loss + args.value_loss_coeff*value_loss
        actionlossarr.append(action_loss)

        critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(critic.parameters(),3)
        critic_optimizer.step()

        if ep%args.update_freq==0:
            actor_optimizer.zero_grad()
            l = torch.cat(actionlossarr).mean()
            l.backward()
            torch.nn.utils.clip_grad_norm(actor.parameters(),3)
            actor_optimizer.step()
            r  =  np.array(rewardarr).sum()/args.update_freq
            print("Episode: {} | Reward: {:.3f}| Length: {}".format(ep,r,ep_len/args.update_freq))
            obsarr = []
            rewardarr = []
            actionlossarr = []
            actionarr = []
            ep_len = 0


        if ep%500==0:
            torch.save(actor.state_dict(),args.save_dir+'/'+args.env_name+'.pt')
            rm,rs,em = test(env,actor,False)
            writer.add_scalar('test/reward_mean',rm,ep)
            writer.add_scalar('test/reward_std',rs,ep)
            writer.add_scalar('test/ep_len_mean',em,ep)
            writer.export_scalars_to_json(args.save_dir+'/'+args.env_name+'_scalars.json')

        writer.add_scalar('train/reward',r,ep)

def test(env,actor,render):
    rew_arr = []
    ep_len_arr = []
    for ep in range(10):
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

    print("Reward Mean: {:.3f}, Std: {:.3f}| Length: {:.3f}".format(
            np.array(rew_arr).mean(),np.array(rew_arr).std(),
            np.array(ep_len_arr).mean()))
    return np.array(rew_arr).mean(),np.array(rew_arr).std(),np.array(ep_len_arr).mean()

if __name__ == '__main__':
    main(sys.argv)
