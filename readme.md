## To train a new agent:
1. Open training_pipeline.py

2. (Optional) Set which config file you want to train with (contains hyperparameters and other configurable options). config_v4.json is a good default.

3. 

## To test a trained agent:
1. Open run_trained_agent.py

2. Set config filename (or leave default v4)

3. Set model_path to the path of your trained policy (should be a zipped file or compressed folder)

4. Run

## To generate a dataset for state prediction (etc.)

You will need to set up a script to run a pretrained policy, and save the state->action transitions to a dataset. 
Note that because the sampled state-action transitions inherently depend on the policy used, you should sample using 
multiple different policies with different behaviors and skill levels. For example, a fully trained policy will tend to
experience good (safe) states, while a checkpoint from early in that policy's training process will tend
to experience bad (unsafe) states.

## Saved models
The saved_models directory contains some pretrained models from a few hyperparameter sweeps. Several of the gamma sweep
models reached high performance and can be used for testing.