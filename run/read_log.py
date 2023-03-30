from lunzi.notebook import *


run = db.runs('literative/wandb/offline-run-20230313_102625-rx64fl5h')[0]

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].set(xlabel='# episodes', ylabel='return')
n_samples, returns = run['step', 'evaluate/eval/mean_policy.return.mean']
axes[0].plot(n_samples // 1000, returns, label='CRABS')
axes[0].legend()

n_trajs, n_unsafe_trajs = run['explore.n_trajs', 'explore.n_unsafe_trajs']
axes[1].set(xlabel='# episodes', ylabel='# unsafe trajectories')
axes[1].plot(n_trajs, n_unsafe_trajs, label='CRABS')
axes[1].legend()

fig.savefig('literative/monitor/')
