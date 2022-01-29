# CFR-RO-TE: Traffic Engineering with Reinforcement Learning in SDN

# Supplement


### Schemes

- Scheme 1: Select critical flows from partial OD flows whose shortest path include centralized links
  - Scheme 1.0: 
    
    trained model were saved at "tf_ckpts/*alpha" folder
    and training logs were saved at "tf_ckpts/*alpha"  folder

    ```python
    # game.py:generate_input

      if config.partial_tm_zeroing == True:
         for z in self.zero_position:
             i, j = z
             self.normalized_traffic_matrices[tm_idx,i,j,h] = 0
  - Scheme 1.1: 
    
    trained model were saved at "tf_ckpts/*alpha+" folder
    and training logs were saved at "tf_ckpts/*alpha+" folder
    
    ```python
    # train.agent
    
     for a in actions:
         if a not in action_space:
             a_batch.append(0)
         else:
             a_batch.append(a)
  
- Scheme 2: Select critical flows from the intersection between cf_potetial and OD flows whose shortest path include centralized links,
        cf_potetial include top K flows calculated by link_load/link_capacity, which k equals to the amount of centralized OD flows
  - Scheme 2.1: 
    
      trained model were saved at "tf_ckpts/*beta" folder
      and training logs were saved at "tf_ckpts/*beta" folder
    
      ```python
       #train.agent

       cf = game.get_critical_topK_flows_beta(config, tm_idx, action_space, critical_links=10)
       for a in actions:
           if a not in cf:
               a_batch.append(0)
           else:
               a_batch.append(a)
### Variable 

- "cf_influence":critical flows are defined by the result of intersection between the shortest path of OD flows and 'critical_link_indexes',
that is, if the result of intersection is larger than 'cf_infuence', it means that one OD flow and critical_link_indexes 
should share at least "cf_infuence+1" links that 
that OD flow can be defined as critical flow. 


    





# CFR-RL: Traffic Engineering with Reinforcement Learning in SDN

This is a Tensorflow implementation of CFR-RL as described in our paper:

Junjie Zhang, Minghao Ye, Zehua Guo, Chen-Yu Yen, H. Jonathan Chao, "[CFR-RL: Traffic Engineering With Reinforcement Learning in SDN](https://arxiv.org/abs/2004.11986)," in IEEE Journal on Selected Areas in Communications, vol. 38, no. 10, pp. 2249-2259, Oct. 2020, doi: 10.1109/JSAC.2020.3000371.

# Prerequisites

- Install prerequisites (test with Ubuntu 20.04, Python 3.8.5, Tensorflow v2.2.0, PuLP 2.3, networkx 2.5, tqdm 4.51.0)
```
python3 setup.py
```

# Training

- To train a policy for a topology, put the topology file (e.g., Abilene) and the traffic matrix file (e.g., AbileneTM) in `data/`, then specify the file name in config.py, i.e., topology_file = 'Abilene' and traffic_file = 'TM', and then run 
```
python3 train.py
```
- Please refer to `data/Abilene` for more details about topology file format. 
- In a traffic matrix file, each line belongs to a N*N traffic matrix, where N is the node number of a topology.
- Please refer to `config.py` for more details about configurations. 

# Testing

- To test the trained policy on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and then run 
```
python3 test.py
```

# Reference

Please cite our paper if you find our paper/code is useful for your work.

@ARTICLE{jzhang,
  author={J. {Zhang} and M. {Ye} and Z. {Guo} and C. -Y. {Yen} and H. J. {Chao}},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={CFR-RL: Traffic Engineering With Reinforcement Learning in SDN}, 
  year={2020},
  volume={38},
  number={10},
  pages={2249-2259},
  doi={10.1109/JSAC.2020.3000371}}
