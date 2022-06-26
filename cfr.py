import numpy as np
import tensorflow as tf

import pyspiel
import time
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

from deep_cfr_tf2 import DeepCFRSolver

if __name__ == '__main__':

    print(tf.config.list_physical_devices())

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    # print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    # game = pyspiel.load_game('kuhn_poker', {"players": 2})

    solver = DeepCFRSolver(
        game,
        path = "/code/data/hunl_fcpa_2p/",
        policy_network_layers=(512, 512, 256, 256, 128, 128, 64, 64, 64),
        advantage_network_layers=(512, 512, 256, 256, 128, 128, 64, 64, 64),
        num_iterations=1,
        num_traversals=5000,
        learning_rate=1e-4,
        batch_size_advantage=1024*10,
        batch_size_strategy=1024*10,
        memory_capacity=int(1e6),
        policy_network_train_steps = 5000,
        advantage_network_train_steps = 1000,
        reinitialize_advantage_networks = True,
        save_advantage_networks = True,
        save_strategy_memories = True,
        save_data_at_end_of_iteration = True,
        infer_device = 'gpu',
        train_device = 'gpu'
    )
    #
    # solver = DeepCFRSolver(
    #     game,
    #     policy_network_layers=(256, 128, 128, 128, 128, 128, 64),
    #     advantage_network_layers=(256, 128, 128, 128, 128, 128, 64),
    #     num_iterations=2,
    #     num_traversals=20000,
    #     learning_rate=1e-4,
    #     batch_size_advantage=20000,
    #     batch_size_strategy=20000,
    #     memory_capacity=int(1e8),
    #     policy_network_train_steps = 5000,
    #     advantage_network_train_steps = 750,
    #     reinitialize_advantage_networks = True,
    #     save_advantage_networks = None,
    #     save_strategy_memories = None,
    #     infer_device = 'cpu',
    #     train_device = 'cpu'
    # )

    # solver.load(True)

    solver.solve()

    # data = solver._get_strategy_dataset()
    # start = time.perf_counter()
    # _, advantage_losses, policy_loss = solver.solve()
    # end = time.perf_counter()

    # for player, losses in list(advantage_losses.items()):
    #     print("Advantage for player:", player,
    #           losses[:2] + ["..."] + losses[-2:])
    #     print("Advantage Buffer Size for player", player,
    #           len(solver.advantage_buffers[player]))
    #
    # print("Final policy loss:", policy_loss)

    # print(f"Took {end - start:0.4f} seconds")
    # conv = exploitability.nash_conv(
    #     game,
    #     policy.tabular_policy_from_callable(game, solver.action_probabilities))
    # print("Deep CFR - NashConv:", conv)