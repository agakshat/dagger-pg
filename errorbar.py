import matplotlib.pyplot as plt
import json

data = json.loads(open('tr2/newLL20/LunarLander-v2_scalars.json').read())
means = [x[2] for x in data['test/reward_mean']]
stds = [x[2] for x in data['test/reward_std']]
eps = [x[1] for x in data['test/reward_std']]
plt.errorbar(eps,means,yerr=stds)
plt.show()
