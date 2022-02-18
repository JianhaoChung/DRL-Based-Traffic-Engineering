# CFR-RO-TE: Traffic Engineering with Reinforcement Learning in SDN



> Athuor: Jianhao Zhong
>
> Organization: School of Computer Science, South China Normal University, Guangzhou, China

# Training

- To train a policy for a topology, put the topology file (e.g., Abilene) and the traffic matrix file (e.g., AbileneTM) in `data/`, then specify the file name in config.py, i.e., topology_file = 'Abilene' and traffic_file = 'TM', and then run 
```shell
python3 train.py
```
- Please refer to `data/Abilene` for more details about topology file format. 
- In a traffic matrix file, each line belongs to a N*N traffic matrix, where N is the node number of a topology.
- Please refer to `config.py` for more details about configurations. 

# Testing

- To test the trained policy on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and then run 
```shell
python3 test.py

python test.py | tee ./result/test_terminal/xxxxx.txt

tensorboard --logdir='./logs/dir'  # visaulize training process

python plotter.py  # visaulize result
```
# Schemes

- **Scheme 1: Select critical flows from partial OD flows whose shortest path include centralized links**

  - Scheme 1.0: 

    trained model and training logs were saved at "tf_ckpts/*alpha" and "log/*alpha" folder respectively.

    ```python
    # game.py:generate_input
    
      if config.partial_tm_zeroing == True:
         for z in self.zero_position:
             i, j = z
             self.normalized_traffic_matrices[tm_idx,i,j,h] = 0
    ```

  - Scheme 1.1: 

    trained model and training logs were saved at "tf_ckpts/*alpha+" and "log/*alpha+" folder respectively.

    ```python
    # train.agent
    
     for a in actions:
         if a not in action_space:
             a_batch.append(0)
         else:
             a_batch.append(a)
    ```

- Scheme 2: 

  - Scheme 2.1: 

    Select critical flows from the intersection between cf_potetial and OD flows whose shortest path include centralized links,
    cf_potetial include top K flows calculated by link_load/link_capacity, which k equals to the amount of centralized OD flows

    trained model and training logs were saved at "tf_ckpts/*beta" and "log/*beta" folder respectively.


  ```python
   #game.py:get_topK_flows_beta
  
   for i in range(self.max_moves*5):
        cf.append(sorted_f[i][0])
  
   #train.agent
  
   cf = game.get_critical_topK_flows_beta(config, tm_idx, action_space, critical_links=10)
   for a in actions:
       if a not in cf:
           a_batch.append(0)
       else:
           a_batch.append(a)
  ```

### Variable 

- "cf_influence": critical flows are defined by the result of intersection between the shortest path of OD flows and 'critical_link_indexes', that is, if the result of intersection is larger than 'cf_infuence', it means that one OD flow and critical_link_indexes  should share at least "cf_infuence+1" links that  that OD flow can be defined as critical flow. 

# Somethings need to know

- mlu means 'maximum link utilization

- max_to_keep: the number of checkpoints to keep. Unless preserved by keep_checkpoint_every_n_hours, checkpoints will be deleted from the active set, oldest first, until only max_to_keep checkpoints remain.  If None, no checkpoints are deleted and everything stays in the active set. Note that max_to_keep = None will keep all checkpoint paths in memory and in the checkpoint state protocol buffer on disk.
  
- TensorFlow objects may contain trackable state, such as tf.Variables, tf.keras.optimizers.Optimizer implementations, tf.data.Dataset iterators, tf.keras.Layer implementations, or tf.keras.Model implementations. These are called trackable objects. A Checkpoint object can be constructed to save either a single or group of trackable objects to a checkpoint file. It maintains a save_counter for numbering checkpoints.

- performacne metric column name 

   ```shell
   col_name = ['tm_idx', 'norm_mlu', 'mlu', 'norm_crit_mlu', 'crit_mlu', 'norm_topk_mlu', 'topk_mlu', 'norm_ecmp_mlu', 'ecmp_mlu', 'optimal_delay/delay', 'optimal_delay/crit_delay', 'optimal_delay/topk_delay', 'optimal_delay/optimal_mlu_delay', 'optimal_delay/ecmp_delay', 'load_multiplier[tm_idx]']
   ```

# Appendix

```python

len(s_batch_agent) == FLAGS.num_iter
len(s_batch) == agent_num * len(s_batch_agent)
```

