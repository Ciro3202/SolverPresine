Presine is an italian imperfect-information card game. Its rules are below. 
This solver tries to achieve super-human play: while this is easily achievable in the small-dimensional rounds, in the bigger rounds it can be particularly 
difficult. 

## Playability
- this repository includes a script that is customizable in order to play heads-up or in 3 players at this game in the CLI. The customizations include number of players, the use of resolver for opponents' strategies (with resolver on, the opponents simulate the specific subgame they are in multiple times to make better decisions), the possibility of showing the probability distributions after your actions to see what the solver would have done in your shoes, "sample" or "greedy" for the realization of the opponents' probability distributions. 
- with "USE_RESOLVER = False" the opponents' strategies are all offline, achieving an immediate response. If the resolver is on, a short computational delay is expected. 
- opponents are trained with information sets and not complete information: they cannot see your cards as you cannot see theirs. 

## Strategies
- for the special round with just 1 card, a deterministic strategy was used.
- for small rounds, complete CFR and MCCFR were used; for middle-sized rounds, a linear model was used to mimic the probability distribution of a resolver; for big rounds, neural networks and online Information-Set Montecarlo Tree Search (IS-MCTS). 
- all strategies were trained using Bocconi's HPC and evaluated in order to achieve a certain level of low exploitability. Exploitability measures how much a fixed strategy can be improved upon by an optimal best response. An exploitability of zero indicates a Nash equilibrium. In this project, exact exploitability was computed only for the smallest heads-up rounds. The remaining results come from approximate best-response evaluations and therefore represent empirical lower bounds, not exact exploitability estimates. 

## Exploitability
These measures are a result of hours of HPC evaluations on single rounds, but they might simply be lower bounds of exploitability (expressed in errors per rounds)
For the HEADS-UP game:
- R1: exact exploitability 0.003525641 errors per round
- R2: exact exploitability 0.020075490 (15,057,744 information states evaluated)
- R3: lower bound exploitability 0.0323 with no resolver
- R4: lower bound exploitability 0.0691 with no resolver 
- R5: lower bound exploitability 0.0504 with no resolver 
For the 3-PLAYERS game:
- R1: deterministic strategy
- R2: lower bound exploitability 0.017383
- R3: lower bound exploitability 0.055 
- R4: lower bound exploitability 0.00417 (this measure is inconclusive: the neural approximate best response needs more training)
- R5: the neural approximate best response needs more training to evaluate correctly the lower bound exploitability

## Game Rules
Presine is a card game for 2–8 players (here playable only in 2-3 players), played with a 40-card Italian deck (Ace, 2–7, Jack, Knight e King with 4 suits). A match has five rounds: R5, R4, R3, R2 and R1, with each player receiving 5, 4, 3, 2 and finally 1 card.
At the beginning of each round, every player predicts how many tricks they will win. The last player cannot make a prediction that would bring the total exactly to the number of tricks available, except in R1.
Players may play any card: there are no suits to follow. The highest card wins the trick, and the winner leads the next one. The strength of the cards are determined by the suits (Clubs is the weakest, then Swords, Cups and Coins which are the strongest) and within the same suit by the value of the card itself (ace is the lowest, king is the highest). 
The Ace of Coins is special: when it is played, its owner chooses whether it is the highest or lowest card in the game.
At the end of the round, your errors are the absolute difference between the number of tricks you predicted and the number you actually won. After all five rounds, the player with the fewest total errors wins.
R1 is played blind: you can see everyone else’s card, but not your own. After all predictions have been made, the cards are revealed and the trick is resolved.
