piefg = []
piefg.append(['V0', 'P1', {'V1': 'Stop', 'V2': 'Go'}])
piefg.append(['V1', 'P2', {'V3': 'Ready', 'V4': 'Go', 'V5': 'Start'}])
piefg.append(['V2', 0, {'V6': 0.6, 'V7': 0.4}])
piefg.append(['V6', 'P2', {'V8': 'E', 'V9': 'F'}])
piefg.append(['V9', 0, {'V10': 0.3, 'V11': 0.7}])
piefg.append(['V11', 'P1', {'V12': 0, 'V13': 1}])

players = ['P1', 'P2']

# Function to extract values for a specific player
def get_player_values(player):
    return tuple(sublist[0] for sublist in piefg if sublist[1] == player)

# Generate results for all players
results = [get_player_values(player) for player in players]

print(results)
