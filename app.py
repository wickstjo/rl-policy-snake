#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import time


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from IPython.display import clear_output


# In[4]:


import ipynb.fs.full.utils as utils


# ### CREATE A SNAKE GAME

# In[5]:


class create_game:
    def __init__(self, dimensions, rewards, display=False):
        
        # CREATE A MAZE, SNAKE & FOOD
        self.create_maze(dimensions)
        self.create_snake()
        self.create_food()
        
        # SET/INIT OTHER VARS
        self.rewards = rewards
        self.display = display
        self.rewards = rewards
        self.score = 0
        self.destroyed = False
        
        # CALCULATE STATE & SHOW THE CURRENT MAZE
        self.calculate_state()
        self.show()
    
    # SHOW THE CURRENT MAZE
    def show(self):
        if self.display:
            
            # CLEAR OLD OUTPUT
            clear_output(wait=True)

            # CREATE & PLOT FIGURE
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(self.maze)
            fig.tight_layout()
            plt.show()
        else:
            pass
        
    # END THE GAME
    def destroy(self):
        self.destroyed = True
        
    # CREATE A MAZE
    def create_maze(self, dimensions):
        self.height = dimensions[0]
        self.width = dimensions[1]
        self.maze = np.zeros((self.height, self.width))
        
    # CREATE FOOD AT A RANDOM POSITION
    def create_food(self):
        
        # FIND ALL EMPTY POSITIONS IN THE MATRIX
        positions = np.argwhere(self.maze == 0)
        
        # PICK A RANDOM OPEN INDEX
        index = random.randint(0, len(positions) - 1)
        
        # SET THE FOODS COORDINATES
        self.food = (positions[index][0], positions[index][1])
        
        # ADD THE FOOD
        self.maze[self.food[0]][self.food[1]] = 2
    
    # CREATE A SNAKE
    def create_snake(self):
        
        # PICK RANDOM COORD
        random_y = random.randint(0, self.height - 1)
        random_x = random.randint(0, self.width - 1)
        
        # CREATE THE SNAKE
        self.snake = [(random_y, random_x)]
        
        # DRAW THE SNAKE
        self.maze[random_y][random_x] = 1
    
    # EVALUATE CURRENT STATE
    def calculate_state(self):
    
        # CREATE TEMP STATE
        self.state = np.zeros(8, dtype=int)
        
        # CURRENT POSITION
        current_y = self.snake[0][0]
        current_x = self.snake[0][1]
        
        # ALL DIRECTIONS
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        # LOOP THROUGH DIRECTIONS AND SET BOUNDRY VALUES
        for index, direction in enumerate(directions):
            self.state[index] = self.check_block(direction)
        
        # SET FOOD DIRECTIONS VALUES
        self.food_direction()
        
    # FETCH NEW POSITIONAL COORDINATE
    def get_position(self, direction):
        next_y = self.snake[0][0]
        next_x = self.snake[0][1]

        # ADD NEXT MOVE DIRECTION
        if direction == 'DOWN':
            next_y += 1
        elif direction == 'UP':
            next_y -= 1
        elif direction == 'RIGHT':
            next_x += 1
        elif direction == 'LEFT':
            next_x -= 1
            
        return (next_y, next_x)
        
    # CHECK BLOCKED PATHS
    def check_block(self, direction):
        
        # GET NEXT COORD POSITION
        next_y, next_x = self.get_position(direction)
        
        # ADD CHECK FOR SNAKEBODY
        if (next_y, next_x) in self.snake:
            return False
        
        # IF Y IS OUT OF BOUNDS
        if next_y < 0 or next_y > self.height - 1:
            return False
        
        # IF X IS OUT OF BOUNDS
        elif next_x < 0 or next_x > self.width - 1:
            return False
        
        else:
            return True
        
    # EVALUATE FOOD DIRECTION - UP RIGHT DOWN LEFT
    def food_direction(self):
        distance = np.array(self.food) - np.array(self.snake[0])
        
        # CHECK & SET EACH DIRECTION
        if distance[0] < 0:
            self.state[4] = 1
        elif distance[0] > 0:
            self.state[6] = 1
        if distance[1] > 0:
            self.state[5] = 1
        elif distance[1] < 0:
            self.state[7] = 1
        
    # CALCULATE STATE VALUE
    def calculate_state_value(self):
        stateNum = 0
        
        for i in range(len(self.state)):
            stateNum += 2**i*self.state[i]
            
        return stateNum
    
    # MOVE THE SNAKE
    def move_snake(self, direction):
        
        # DEFAULT TO ZERO REWARD
        reward = 0

        # GET THE NEW POSITION
        next_y, next_x = self.get_position(direction)

        # THE THE SNAKE HITS ITSELF
        if ((next_y, next_x) in self.snake) or (next_y < 0 or next_y > self.height - 1) or (next_x < 0 or next_x > self.width - 1):
            self.destroy()
            reward = self.rewards['death']

        # EAT FOOD & GROW
        elif (self.maze[next_y][next_x] == 2):

            # GROW THE SNAKE BY UPDATING THE HEADS POSITION
            self.snake.insert(0, (next_y, next_x))
            self.maze[next_y][next_x] = 1

            # INCREASE THE SCORE & CREATE NEW FOOD
            self.score += 1
            self.create_food()
            
            # SET REWARD
            reward = self.rewards['grow']

            # SHOW UPDATED MAZE
            self.show()

        # OTHERWISE, MOVE
        else:
            
            # SET REWARD IF SNAKE MOVES CLOSET TO THE FOOD
            if (direction == 'DOWN' and self.state[4:][2] == 1) or (direction == 'UP' and self.state[4:][0] == 1) or (direction == 'RIGHT' and self.state[4:][1] == 1) or (direction == 'LEFT' and self.state[4:][3] == 1):
                reward = self.rewards['direction']

            # MOVE SNAKE HEAD
            self.snake.insert(0, (next_y, next_x))

            # REMOVE SNAKE TAIL
            tail = self.snake.pop()

            # RENDER NEW HEAD & REMOVE OLD TAIL
            self.maze[next_y][next_x] = 1
            self.maze[tail[0]][tail[1]] = 0

            # SHOW UPDATED MAZE
            self.show()
        
        # CALCULATE NEW STATE
        self.calculate_state()
        
        return self.calculate_state_value(), reward, self.score


# ### EVALUATE POLICY

# In[6]:


def evaluate(actions, rewards, epoch, best, policy, maze_size, iterations, max_steps, delay, display):
    
    # SCORE HISTORY
    scores = []
    
    # PLAY X AMOUNT OF GAMES
    for index in range(iterations):
        
        # INSTANTIATE VARS
        game = create_game(maze_size, rewards, display=display)
        state = game.calculate_state_value()
        score, old_score, steps = 0, 0, 0

        # WHILE THE GAME IS NOT DESTROYED
        while not game.destroyed:
            
            # CHECK POLICY OPTIONS & SELECT AN ACTION
            options = policy[state, :]
            action = actions[np.argmax(options)]

            # PERFORM THE ACTION
            state, reward, score = game.move_snake(action)
            
            # INCREMENT STEP COUNTER IF OLD & NEW SCORES MATCH
            if score == old_score:
                steps += 1
                
            # OTHERWISE, RESET STEPS & OLD SCORE
            else:
                old_score = score
                steps = 0
                
            # IF THE MAXIMUM STEPCOUNT IS EXCEEDED, BREAK THE LOOP
            if steps >= max_steps:
                break

            # SLEEP FOR...
            if display:
                time.sleep(delay)
                
        # APPEND TO SCORES
        scores.append(score)
    
    return np.average(scores), scores


# ### TRAINING

# In[7]:


def training(gamma, epsilon, rewards, epochs, evaluation, maze_size, max_steps, delay, display=False):
    
    # POSSIBLE ACTIONS
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # CREATE A POLICY CONTAINERS
    state_count = 2 ** 8
    policy = np.zeros((state_count, len(actions)))
    pair_policy = np.zeros([epochs, state_count, len(actions)])
    
    # BEST RESULTS
    best_score = 0
    best_policy = []
    epoch_scores = []
    
    # LOOP THROUGH EACH EPOCH
    for epoch in range(epochs):
        
        # INSTANTIATE GAME VARS
        game = create_game(maze_size, rewards, display=display)
        state = game.calculate_state_value()
        score = 0
        
        # WHILE THE GAME IS NOT DESTROYED
        while not game.destroyed:
            
            # GENERATE A RANDOM NUMBER
            rand_num = random.uniform(0, 1)
            
            # IF IT WAS WITHIN EPSILON RANGE, CHOOSE ACTION RANDOMLY
            if rand_num < epsilon['value']:
                action = random.randint(0, len(actions) - 1)
                
            # OTHERWISE, SELECT IT FROM THE POLICY
            else:
                options = policy[state, :]
                action = np.argmax(options)
                
            # PERFORM THE ACTION
            new_state, reward, score = game.move_snake(actions[action])
            
            # SET STATE ACTION PAIR IN POLICY
            policy[state, action] = reward + gamma * np.max(policy[new_state, :])
            
            # REPLACE OLD STATE
            state = new_state
            
            # SLEEP FOR...
            if display:
                time.sleep(delay)
            
        # IF SELECTED, DECAY EPSILON MULTIPLICATIVELY
        if epsilon['decay']:
            epsilon['value'] = epsilon['value'] - (epsilon['value'] / epochs)
            #print(epsilon)
        
        # APPEND TO EPOCH SCORES
        epoch_scores.append(game.score)
            
        # COPY THE POLICY TO THE PAIR CONTAINER
        pair_policy[epoch, :, :] = np.copy(policy)
        
        # EVERY EPOCH BREAKPOINT DO...
        if epoch % evaluation['breakpoint'] == 0:
            
            # EVALUATE THE POLICY
            average_score, scores = evaluate(**{
                'actions': actions,
                'rewards': rewards,
                'epoch': epoch,
                'best': best_score,
                'policy': policy,
                'maze_size': maze_size,
                'iterations': evaluation['epochs'],
                'max_steps': max_steps,
                'delay': delay,
                'display': display
            })
            
            # IF A BETTER SCORE IS FOUND, MARK IT DOWN
            if average_score > best_score:
                best_score = average_score
                best_policy = np.copy(policy)
            
        # IF DISPLAY IS TURNED OFF, SHOW EPOCH COUNT
        if epoch % 5000 == 0:
            if not display:
                clear_output(wait=True)
                print('EPOCH {} REACHED'.format(epoch))
        
    return best_score, best_policy, epoch_scores


# ### EXECUTE

# In[8]:


policy_score, policy, epoch_scores = training(**{
    'gamma': 0.2,
    'epsilon': {
        'value': 0.3,
        'decay': True
    },
    'rewards': {
        'death': -100,
        'direction': 1,
        'grow': 30
    },
    'epochs': 100001,
    'evaluation': {
        'breakpoint': 50,
        'epochs': 25
    },
    'maze_size': [10, 10],
    'max_steps': 100,
    'delay': 0.0001,
    'display': False
})


# In[ ]:


utils.save('script-test', policy)


# In[ ]:




