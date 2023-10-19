import json

# Read the network.json file
with open("network.json", "r") as f:
    network = json.load(f)

# Create a set to store all pairs (a, b) such that a follows b but b doesn't follow back a
pairs = set()

# Iterate over all users in the network
for user, data in network.items():
    # Get the list of users that the current user follows
    following = data.get("Connections", [])

    # Iterate over all users that the current user follows
    for follower in following:
        # Check if the follower doesn't follow back the current user
        if user not in network.get(str(follower), {}).get("Connections", []):
            pairs.add((user, str(follower)))

# Remove mutual follows from the set
mutual_follows = {(b, a) for (a, b) in pairs if (b, a) in pairs}
pairs -= mutual_follows

# Print all pairs (a, b) such that a follows b but b doesn't follow back a
if len(pairs) > 0:
    for pair in pairs:
        print(pair)

    # Print the number of such pairs
    print(f"Number of pairs where 'a' follows 'b' but 'b' doesn't follow back 'a': {len(pairs)}")
else:
    # If there are no such pairs, display that the relationship is completely bidirectional
    print("The relationship is completely bidirectional.")
