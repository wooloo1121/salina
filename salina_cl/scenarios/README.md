# CRL Scenarios

## Description

A CRL Scenario is a sequence of training and testing tasks. Each task is associated with:
* a SaLinA agent that corresponds to the environment (with auto-reset)
* Additional informations, and particularly the `n_interactions` value that represents the number of environment steps allowed for the task. Note that this is the role of the algorithm  to take into account this maximum number of interactions. 

## Provided Scenarios

You can add your own scenario in this repo. To use them in an experiment, simply add a yaml file `my_scenario` in the [configs/scenario](configs/scenario/) folder and use the option `scenario=my_scenario` in the command line. Here is a list of the current scenarios:

*Debbuging scenarios:*
* `cartpole_debug`
* `halfcheetah_debug`

*'Simple' scenarios:*
* `cartpole_7tasks`: 7 tasks with varying parameters
* `halfcheetah_simple`: 10 HalfCheetah environments with a linear increasing gravity coefficient (from 0.2 to 2)

*'Hard' scenarios:*
* `halfcheetah_hard`: in progress ...
