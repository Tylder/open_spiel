## Game Requirements to use CFR

List of stuff that the game environment must be able to do or give info on.
Basically its stuff I need to make sure works before I can run CFR on a custom game.

- Immutable state, i.e. when state changes it creates a new state object and does not modify the previous.



### Methods

- newInitialState() -> new fresh state at the start of game
- numPlayers() -> number of players in game, used for iteration
- 