def to_sfg(piefg):
    def payoff_for_players(strategy_profile, curr_node, which_strat):
        for sublist in piefg:
            if sublist[0] == curr_node:
                # If the node contains the payoff directly
                if len(sublist) > 1 and isinstance(sublist[1], list):
                    return sublist[1]

                # If Nature's turn, process probabilities
                if len(sublist) > 1 and sublist[1] == 0:
                    return payoff_for_nature(strategy_profile, curr_node, which_strat, [0, 0])

                # Process player strategy
                if len(sublist) > 2:
                    strategy_dict = sublist[2]
                    player_idx = players.index(sublist[1])
                    desired_strategy = strategy_profile[player_idx][which_strat]
                    which_strat += 1

                    # Reverse-map strategy to node
                    key_for_desired_strategy = next(
                        (key for key, value in strategy_dict.items() if value == desired_strategy),
                        None,
                    )
                    if not key_for_desired_strategy:
                        raise ValueError(
                            f"Invalid strategy '{desired_strategy}' at node '{curr_node}'."
                        )
                    return payoff_for_players(strategy_profile, key_for_desired_strategy, which_strat)

        raise ValueError(f"Node '{curr_node}' not found in the game tree.")

    def payoff_for_nature(strategy_profile, curr_node, which_strat, payoff):
        for sublist in piefg:
            if sublist[0] == curr_node:
                if len(sublist) > 2:
                    strategy_dict = sublist[2]
                    total_payoff = [0, 0]
                    for key, prob in strategy_dict.items():
                        next_node = key
                        nature_payoff = payoff_for_players(strategy_profile, next_node, which_strat)
                        total_payoff = [
                            total_payoff[i] + prob * nature_payoff[i]
                            for i in range(len(nature_payoff))
                        ]
                    return total_payoff

        raise ValueError(f"Nature's node '{curr_node}' not found in the game tree.")

    # Extract players from the first element in piefg
    players = piefg[0]

    # Extract strategies for players
    playerstratlist = [[] for _ in range(len(players))]
    for sublist in piefg:
        if len(sublist) > 2 and isinstance(sublist[2], dict):
            for i, player in enumerate(players):
                if sublist[1] == player:
                    playerstratlist[i].append(tuple(sublist[2].values()))

    # Cartesian product of strategies for all players
    strategy_profiles = list(product(*[list(product(*strats)) for strats in playerstratlist]))

    U = {}
    is_nature_at_root = any(sublist[0] == "V0" and len(sublist) > 1 and sublist[1] == 0 for sublist in piefg)

    for profile in strategy_profiles:
        curr_node = "V0"
        which_strat = 0
        if is_nature_at_root:
            payoff = payoff_for_nature(profile, curr_node, which_strat, [0, 0])
        else:
            payoff = payoff_for_players(profile, curr_node, which_strat)
        U[profile] = payoff

    return U
