This repository is the companion to the blog post here: https://lars-chen.github.io/rl-blog/learning-barrier-certificates/

# -Installation- 
1. First navigate to the directory you house CRABS in. Create a new Anaconda environment with python 3.9 to house the project with the right versions of software.
   > conda create --name safe --file requirements.txt

2. Next download the appropriate version (Linux, Mac, etc) version of Mujoco 210 from https://github.com/deepmind/mujoco/releases and put it into your file structure as such: <home folder>/.mujoco/mujoco210
After downloading it, you will need to add it to Environmental Variables. 
   > export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<home folder>/.mujoco/mujoco210/bin

3. Follow the instructions on https://docs.wandb.ai/quickstart to set up a wandb account. If you wish to run it without, you can run 
   > wandb disabled 

or you can run wandb offline and upload runs later 
     > wandb offline

4. Run `export PYTHONPATH=$PYTHONPATH:.`

# -Running CRABS-

To do a full run of CRABS, one needs to do four steps: 1) pretrain a model with Safe-Actor-Critic, 2) pretrain a dynamics model, 3) pretrain the barrier certificate (h), and finally 4) train the algorithm. Below we detail how to run each step. We found training times on a GPU for each step were respectively roughly 12-18 hours, 1-2 hours, 1-2 hours, and 1.5+ days.
For steps 2-4, the program will be run by editing a config file (found in configs folder). 
The config file lets you switch between environments, change thresholds, learning rates, and upload checkpoints to begin your next learning task.
There are checkpoints for the three tasks we cover in the blog post in the ckpts folder.


1) Run mf_td3.py with your chosen environment as a command line variable --env.id, for example
   > python run/mf_td3.py --env.id SafeHopper-v0
2) In the config file, change task to at the bottom of the file to pretrain_model, as such 'task: "pretrain_model”'. An initial policy checkpoint from the previous task is required.
   Run the below command, where it passes a working directory and the config file
   > python run/main.py --root_dir /tmp/crabs/pretrain/ -c ./configs/hopper/Hopper-pretrain_model.json5
3) In the config file, change task to “pretrain_h”. An initial policy and model checkpoint are required.
   > python run/main.py --root_dir /tmp/crabs/pretrain/ -c ./configs/hopper/Hopper-pretrain_h.json5
4) In the config file, change task to “train_policy”. An initial policy, model, and h are required. 
   > python run/main.py --root_dir /tmp/crabs/pretrain/ -c ./configs/hopper/Hopper-train_policy.json5
