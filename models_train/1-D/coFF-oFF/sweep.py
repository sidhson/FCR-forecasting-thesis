import wandb
import os, sys
cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from train_model import train

### NOTE: Insert SWEEP ID of current run.

SWEEP_ID = 'g_and_f/coFF-oFF-project/p0uc124q'

if __name__ == '__main__':
    '''
    Create Sweep job from CLI: 
        wandb sweep sweep_config.yaml 

    Insert the SWEEP_ID from the job created.
    
    Run this script to create an agent to run a job 
    with hyperparameters from the Sweep job. 
    '''
    wandb.agent(sweep_id=SWEEP_ID, function=train)