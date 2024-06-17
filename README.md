## Chess Opening Recommendations
This project provides chess opening recommendations based on the games of a given Lichess user. It analyzes the user's games, extracts the openings played, and recommends similar openings using a graph-based relatedness network.

![image](https://github.com/shayvana/openingrecs/assets/19787070/a7890b6e-49fb-4668-ae99-5a27295197b3)

### Features
* Fetch chess games of a Lichess user.
* Extract and normalize chess openings from the games.
* Recommend similar chess openings based on a relatedness network.

### Usage
* Enter a Lichess username on the homepage.
* Click "Get Recommendations".
* View the recommended chess openings along with the explanation.

### Building the Bipartite Network
The build_bipartite_network function reads PGN files and builds a bipartite network using NetworkX. It adds nodes for players and openings and creates edges between players and their played openings.

### Projecting and Filtering the Network
The project_and_filter_network function projects the bipartite network to a unipartite network of openings using a co-occurrence matrix. It then filters the network using the bipartite configuration model (bicm) to retain significant edges.

### Making Recommendations
The recommend_openings function compares the normalized user openings with the relatedness network. It calculates scores based on the weights of related openings and provides the top recommendations with explanations.
