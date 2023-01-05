*## Intro
RL is teaching a software agent how to behave in an enviornment by telling it how good its doing. 
***
## Deep Q Learning
Uses Deep Neural Networks
***
## Steps-
* Game(Pygame)-->	
	play_step(action)->reward,game_over,score
	
* Agent
	-game
	-model
	Training =:
	-state = get_state(game)
	-action - get_move (model.predict)
	-reward,game_over,score = game.play_step(action)
	-new_state = get_state(game)
	-remember
	-model.train()
* Model
	Linear_QNet(DQN)
	-model.predict(state)
	->action
***
## Reward
	-eat food +10
	-game over -10
	-else 0
***
## Action
	[1,0,0]-> straight
	[0,1,0]-> right turn
	[0,0,1]-> left turn
***
## State
	[danger - s,r,l,
	direction - s,r,l,d
	food - s,r,l,d
	]

![Screenshot from 2023-01-04 14-24-38.png](:/995ce517268540f1bc4076b5f17ddbeb)
***
## Model
State(11) - Feed Forward Neural Network- Action(3)
***
## Deep Q Learning
Q Learning
Q value = Quality of action
0. Init Q (=init model)
1. Choose action (model.predict(state))/random
2. Perform Action
3. Measure Reward
4. Update Q Value (+ train model)
5. Go to 1
***
## Bellman Equation
NewQ(s,a) = Q(s,a) + $\alpha$*[R(s,a)+$\gamma$*maxQ'(s',a') - Q(s,a)]
$\gamma$ = discount_rate
$\alpha$ = learning_rate
***
SImplified Version-
Q = model.predict(state_0)
Q_new = R+dr*max(Q(state_1))

loss = (Q_new-Q)^2
***
# Game
(Optional - virtual env)
install pygame, tensorflow,ipython,matplotlib
***
TODO:
* Reset
* Reward
* Play(action)->direction
* Game_iteration
* Is_Collision
***
Importing Stuff
`import pygame`
`import random`
`from enum import Enum`
`from collections import namedtuple`
(Add Font?)
***
Make Direction Enum
`class Direction(Enum):`
`RIGHT = 1`
`LEFT = 2` and so on
***
(function) namedtuple(typename: str, field_names: str | Iterable[str], *, rename: bool = ..., module: str | None = ..., defaults: Iterable | None = ...) -> Type[tuple]
Returns a new subclass of tuple with named fields.
`Point = namedTuple('Point','x,y')`
***
Next Step -  Make Snake Game
Done - Move Back to RL
***
## Make Reset Function
def reset(self):
        
        #init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,Point(self.head.x-BLOCK_SIZE,self.head.y),Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
				self.iteration = 0
***
## Add Action Functionality
## Add Reward Values
## Check if nothing changes for a long time

Game is ready, takes action and returns state.
Game Done.
***
# Agent Time
import torch, numpy and snake game stuff
define
MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.001
***
Begin Agent Class
Define Functions
def __init__(self):
        pass
    
    def get_state(self,game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass
    
    def get_action(self,state):
        pass
		
		
		Define all functions and use deque to store state
***
# Model

def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
	
	    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)
	
	learn pytorch later
	
***
# Trainer
Done with notes
