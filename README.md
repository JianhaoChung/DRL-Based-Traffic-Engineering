# PKE-DRL: Prior Knowledge Enhanced Deep Reinforcement Learning Approach for Network Rerouting

> Athuor:  Zhong Jianhao, YeRuihao,  Zheng Weiping,  Zhao Ganshen
>
> Organization: School of Computer Science, South China Normal University, Guangzhou, China

## create enviroment

```bash
# conda env export --file ./enviroment.yaml

conda env create -f ./enviroment.yaml
```

## configuration:

```python

# config.py #

scheme # DRL-Based rerouting scheme
# - baseline: CFR-RL
# - alpha_update: PKE-DRL

max_moves # 挑选多少百分比例的流量进行重路

central_flow_sampling_ratio # 混合流量采样存在多少百分比例的中心流

scheme_explore 
# - None 表示不进行混合采样
# - lastK_centralized_sample 表示混合采样方案，采样中心流和弱中心流

```

## train

To train a policy for a topology, put the topology file (e.g., Abilene) and the traffic matrix file (e.g., AbileneTM) in `data/`, then specify the file name in config.py, i.e., topology_file = 'Abilene' and traffic_file = 'TM', and then run 

```shell
python train.py
```

- Please refer to `data/Abilene` for more details about topology file format. 
- In a traffic matrix file, each line belongs to a N*N traffic matrix, where N is the node number of a topology.
- Please refer to `config.py` for more details about configurations. 

###### visaulize training process

```shell
tensorboard --logdir='./logs/dir' 
# for example
tensorboard --logdir='./logs/TE_v2-CFR-RL_pure_policy_Conv_Ebone_TM_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves15'
```

## test

To test the trained policy on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and then run：

```bash
python test.py
# python test.py | tee ./result/test_terminal/xxxxx.txt

```

## plotter



```bash
# plotter.py | plotter_utils.py #
```



