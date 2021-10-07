import numpy as np
import tensorflow as tf

import pyspiel

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
        policy_network_layers=(64, ),
        advantage_network_layers=(64, ),
        num_iterations=1,
        num_traversals=10,
        learning_rate=1e-4,
        batch_size_advantage=256,
        batch_size_strategy=256,
        memory_capacity=int(1e8),
        policy_network_train_steps = 100,
        advantage_network_train_steps = 100,
        reinitialize_advantage_networks = True,
        save_advantage_networks = "/code/data/advantage_nets/hunl_fcpa_2p",
        save_strategy_memories = "/code/data/strategy_memories/hunl_fcpa_2p",
        infer_device = 'cpu',
        train_device = 'cpu'
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

    _, advantage_losses, policy_loss = solver.solve()

    for player, losses in list(advantage_losses.items()):
        print("Advantage for player:", player,
              losses[:2] + ["..."] + losses[-2:])
        print("Advantage Buffer Size for player", player,
              len(solver.advantage_buffers[player]))
    print("Strategy Buffer Size:",
          len(solver.strategy_buffer))
    print("Final policy loss:", policy_loss)
    # conv = exploitability.nash_conv(
    #     game,
    #     policy.tabular_policy_from_callable(game, solver.action_probabilities))
    # print("Deep CFR - NashConv:", conv)