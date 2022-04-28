# Visualize Training Log of Advantage,Reward,Entrophy,Learning-Rate
tensorboard --logdir='./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha+_lastK_sample_0.4scaleK_maxMoves15'
tensorboard --logdir='./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves15'
tensorboard --logdir='./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_baseline_maxMoves15'

tensorboard --logdir_spec=name1:/path/to/logs/1,name2:/path/to/logs/2

tensorboard --logdir_spec='./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha_update_lastK_centralized_sample_0.7scaleK_maxMoves10', './logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha_update_lastK_centralized_sample_0.7scaleK_maxMoves10'

tensorboard --logdir_spec='Model1':'./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha_update_lastK_centralized_sample_0.7scaleK_maxMoves10', 'Model2':'./logs/TE_v2-CFR-RL_pure_policy_Conv_Abilene_TM_alpha_update_lastK_centralized_sample_0.7scaleK_maxMoves10'

python test.py tee |  tee ./result/test_terminal/pure-policy-conv-alpha+-lastK-sample-0.5scaleK-maxMoves30-ckpt70.txt
python plotter.py | tee ./result/img/Policy_Alpha_LKC_0.25K_maxmoves15_and_Alpha+_LK_0.5K_maxmoves15_report.txt
