from plotter_utils import *

from config import get_config
from absl import flags

FLAGS = flags.FLAGS
config = get_config(FLAGS)

if __name__ == '__main__':

    convergence_time_plot(topo='Abilene', save_plot=save_plot)
    convergence_time_plot(topo='Ebone', save_plot=save_plot)

    # test_mode = 'ebone'
    test_mode = None
    save_plot = True
    maxmoves_exp = [10, 15, 20]
    maxmoves = maxmoves_exp[0]

    if test_mode == 'ebone':
        efile1 = ebone_data('PKE-DRL', max_moves=maxmoves)
        efile2 = ebone_data('CFR-RL', max_moves=maxmoves)
        ebone_data_analyzer(efile1, efile2, label_name=config.label_name, config=config, save_plot=save_plot)
        exit(1)

    file1 = abilene_data(model='PKE-DRL', max_moves=maxmoves, pke_drl_idx=0)
    file2 = abilene_data(model='CFR-RL', max_moves=maxmoves)

    if os.path.exists(file1) and os.path.exists(file2):
        data_analyzer(file1, file2, label_name=config.label_name, save_plot=save_plot,
                      avg_week_plot=True, avg_day_plot=False)
        pr_week_detail_plot(file1, file2, scheme='mlu', label_name=config.label_name, save_plot=save_plot)
        pr_week_detail_plot(file1, file2, scheme='delay', label_name=config.label_name, save_plot=save_plot)

    pr_plot_abilene(config.label_name, save_plot=save_plot)
    pr_plot_ebone(config.label_name, save_plot=save_plot)

    flow_sampling_effect_pr_plot('mlu', save_plot=save_plot)




