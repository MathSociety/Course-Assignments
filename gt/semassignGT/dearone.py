import numpy as np
import scipy
import matplotlib.pyplot as plt
from itertools import product

def to_sfg(piefg):

    def payoff_for_players(strategy_profile, curr_node, which_strat):

        # Search in piefg for curr_node
        for sublist in piefg:

            if sublist[0] == curr_node:  # Check if the first element is the curr_node (e.g., 'V0', 'V1', etc.)
                # If we find a valid final node (payoff node)
                if isinstance(sublist[1], list):
                    return sublist[1]  # Return the final payoff list

                # Find the corresponding player (index of the player associated with curr_node)
                # player_idx = players.index(sublist[1])  # sublist[1] should be the player ('P1', 'P2', etc.)
                    
                # If player is 0 (Nature), call the payoff_for_nature function
                if sublist[1] == 0:
                    payoff = payoff_for_nature(strategy_profile, curr_node, which_strat + 1, 0)
                    return payoff  # Return the payoff for player 0 (Nature)

                # Retrieve the dictionary for that player and strategy
                strategy_dict = sublist[2]  # The dictionary with the strategy values (e.g., {'V1': 'Stop', 'V2': 'Go'})

                # If the strategy is a valid dictionary, find the corresponding key (strategy)
                if isinstance(strategy_dict, dict):
                    player_idx = players.index(sublist[1])
                    desired_strategy = strategy_profile[player_idx][which_strat]  # Extract the key from the dictionary
                    which_strat+=1
                    #reverse dictionary
                    key_for_desired_strategy=next((key for key,value in strategy_dict.items() if value==desired_strategy),None)
                    curr_node = key_for_desired_strategy  # Update curr_node to the new key
                    which_strat=0


                else:
                    raise Exception(f"Unexpected structure: {sublist[1]} should be a string and {sublist[2]} should be a list.")


    def payoff_for_nature(strategy_profile, curr_node, which_strat, payoff):
        """
        Function to calculate the payoff for Nature.
        """
        # 1. Find the number of strategies of curr_node
        c0 = 0
        strategy_dict = {}
        # Search in piefg for the current node (curr_node)
        p=[0]* len(players)
        for sublist in piefg:
            if sublist[0] == curr_node:  # Check if the first element is the curr_node
                strategy_dict = sublist[2]  # The strategy dictionary is the 3rd element in the sublist
                if isinstance(strategy_dict, dict):
                    c0 = len(strategy_dict)  # Number of strategies is the number of keys in the dictionary
                break


        # 2. Iterate over the strategies for curr_node
        for i in range(c0):
            # Get the strategy corresponding to the current index (i)
            strategy_keys = list(strategy_dict.keys())
            selected_key = strategy_keys[i]

            # 3. Update the curr_node to be the key of strategy[i] (which will be a probability)
            curr_node = selected_key  # This is the new curr_node based on the strategy selected
            # 4. Search for the updated curr_node in piefg
            for sublist in piefg:
                if sublist[0] == curr_node:  # Match the current node (which is now a strategy)
                    # Check if the current node corresponds to a player (i.e., not Nature player)
                    if (isinstance(sublist[1], str) and sublist[1] != 0) or isinstance(sublist[1],list):  # Checking if the node corresponds to a player
                        # Call payoff_for_players for this player
                        p_player = payoff_for_players(strategy_profile, curr_node, which_strat)
                        p = [p[i] + (strategy_dict[selected_key] * p_player[i]) for i in range(len(p_player))] # Multiply the strategy probability by the payoff and accumulate
                    else:
                        # If Nature, call payoff_for_nature
                        p = payoff_for_nature(strategy_profile, curr_node, which_strat, p)
                    break

        return p
    


    # Extract sublists with dictionaries (where strategies are defined)
    list_with_dicts = [
        sublist for sublist in piefg
        if any(isinstance(item, dict) for item in sublist)
    ]

    # Extract players (non-dict, non-list sublists)
    # Extract players as the first sublist from piefg
    for sublist in piefg:
        if isinstance(sublist, list) and all(isinstance(item, str) for item in sublist):
            players = sublist
            break

    # List of strategies for each player
    playerstratlist = [[] for _ in range(len(players))]

    # Collect strategies for each player
    for i in range(len(players)):
        playerstratlist[i] = [
            sublist for sublist in list_with_dicts if players[i] in sublist
        ]
    
    # Build the strategies for each player
    playerstrat = []
    for player_strategies in playerstratlist:
        strategies = tuple(tuple(d.values()) for _, _, d in player_strategies)

        # Handle case where there is only one strategy, avoid wrapping it in a tuple
        if len(strategies) == 1:
            playerstrat.append((strategies[0],))  # Ensure it's a tuple for consistency
        else:
            playerstrat.append(strategies)

    playerstrat = tuple(playerstrat)
    # Compute the cross-product of strategies if needed
    updated_playerstrat = []
    for strategies in playerstrat:
        if len(strategies) > 1:
            cross_product = tuple(product(*strategies))
            updated_playerstrat.append(cross_product)
        else:
            updated_playerstrat.append(strategies)
    
    updated_playerstrat = tuple(updated_playerstrat)

    # Compute all strategy profiles (Cartesian product of all player strategies)
    strategy_profiles = list(product(*updated_playerstrat))


    # Initialize payoff dictionary U
    U = {}

    # Determine if Nature (player 0) is at the root (V0)
    is_nature_at_root = False
    for sublist in piefg:
        if sublist[0] == 'V0' and sublist[1] == 0:  # If player 0 (Nature) is at V0
            is_nature_at_root = True
            break

    # Loop through each strategy profile and compute the payoff
    for profile in strategy_profiles:
        # Set the first player at V0
        curr_node = 'V0'
        
        # Set the strategy index (start with the first strategy in the profile)
        which_strat = 0  # This can be adjusted dynamically if needed

        # Check if Nature is at root, then call payoff_for_nature for each profile
        if is_nature_at_root:
            payoff = payoff_for_nature(profile, curr_node, which_strat, [0,0])
        else:
            payoff = payoff_for_players(profile, curr_node, which_strat)
        
        # Store the payoff in dictionary U
        U[profile] = payoff

    # Return the dictionary U containing the payoffs for each strategy profile
    # Get the player values from piefg
    results = [tuple(sublist[0] for sublist in piefg if sublist[1] == player) for player in players]

    # Remove the first element from the second subtuple if it's 'P1'
    results[1] = results[1][1:]  # Slice to remove the first element of the second subtuple

    # Convert results into a tuple
    results = tuple(results)

    # Add the dictionary U to the results
    results = list(results)
    results.append(U)

    # Convert back to a tuple if needed
    results = tuple(results)

    return results


# Example game tree
piefg = []
piefg.append(['P1', 'P2'])
piefg.append('V0')

piefg.append(['V0', 'P1', {'V1': 'Stop', 'V2': 'Go'}])
piefg.append(['V1', 'P2', {'V3': 'Ready', 'V4': 'Go', 'V5': 'Start'}])
piefg.append(['V2', 0, {'V6': 0.6, 'V7': 0.4}])
piefg.append(['V6', 'P2', {'V8': 'E', 'V9': 'F'}])
piefg.append(['V9', 0, {'V10': 0.3, 'V11': 0.7}])
piefg.append(['V11', 'P1', {'V12': 0, 'V13': 1}])


piefg.append(['V3', [3, 8]])
piefg.append(['V4', [8, 10]])
piefg.append(['V5', [6, 3]])
piefg.append(['V7', [5, 5]])
piefg.append(['V8', [5, 5]])
piefg.append(['V10', [6, 7]])
piefg.append(['V12', [2, 3]])
piefg.append(['V13', [1, 0]])

sfg = to_sfg(piefg)
print(sfg)