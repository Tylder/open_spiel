## Game Requirements to use CFR

List of stuff that the game environment must be able to do or give info on.
Basically its stuff I need to make sure works before I can run CFR on a custom game.

- Immutable state, i.e. when state changes it creates a new state object and does not modify the previous.



### Methods 

- newInitialState() -> new fresh state at the start of game
- is_terminal() -> bool, true if we are at the end state, should be able to get returns/rewards at this state
- 
- numPlayers() -> number of players in game, used for iteration


### State methods
- currentPlayer() -> returns the player to act
- informationStateTensor() -> tensor which holds the information for the state, used by ml algo to as the feature
- numActions() -> number of total available actions, in total
- legalActions() -> list of actions available
- legalActionsMask() -> numpy 1d array of 1/0 with the same length as actions to mask out available actions