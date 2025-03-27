# Value Iteration Package


## Installation

To install the package, use the following line of code.


```python
pip install git+https://github.com/bpowers402/Value_it
```

    Collecting git+https://github.com/bpowers402/Value_itNote: you may need to restart the kernel to use updated packages.
    
      Cloning https://github.com/bpowers402/Value_it to c:\users\powersb\appdata\local\temp\pip-req-build-hzw0gase
      Resolved https://github.com/bpowers402/Value_it to commit 4a8d695854f8e20a6e22e0e64aff486ba92aa5ee
      Installing build dependencies: started
      Installing build dependencies: finished with status 'done'
      Getting requirements to build wheel: started
      Getting requirements to build wheel: finished with status 'done'
      Preparing metadata (pyproject.toml): started
      Preparing metadata (pyproject.toml): finished with status 'done'
    Requirement already satisfied: numpy in c:\users\powersb\stor-601\env\lib\site-packages (from value_it==1.0.0) (2.2.4)
    

      Running command git clone --filter=blob:none --quiet https://github.com/bpowers402/Value_it 'C:\Users\powersb\AppData\Local\Temp\pip-req-build-hzw0gase'
    

## Using the value_it function

The value_it function can take 6 inputs. The first four are always required. These are the state space, the action space, the probability matricies, and the reward matrix. The final two inputs are the number of iterations required and the discount factor $\gamma$. If no input is given for these two variables, the default value will be taken as $1000$ iterations, and $\gamma = 0.8$. 

The expected format for the inputs is as follows:
- For the state space and action space, inputs should be vectors
- For the state transition function, the expected input is a numpy array which repreresents a $s \times s \times a$ matrix, where $s$ is the number of states and $a$ is the number of actions. This can be thought of as given the trasition probability matrix for every state. 
- The Rewards input should be an $s \times a$ matrix (again, $s$) is the number of states, and $a$ is the number of actions, and the values in the matrix should be the expected reward of taking the corresponding action given the state you are in.
- The number of iterations, $I$ and the discount factor $\gamma$ should be integer inputs. 

## Example

The first example given is the same as exercise 9.27 from https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27. For this example, we have two possible states and two actions. The first step in implementing the method is to import numpy, as we will need the probability and reward arrays to be inputted using numpy.


```python
import numpy as np
from value_it import value_it
```

Suppose the problem has the two states: "healthy" and "sick", and two actions: "relax" and "party".  The inputs of the matrices mentioned are assumed to be given in this order, so the top left entry of the reward matrix would be the expected reward of relaxing given that you were healthy. The reward matrix in this example is 
Matrix R:

| 7  | 10 |
|----|----|
| 0  | 2  |

The state transition matrix for entering the healthy state is given by:
$$
P = 
\left(\begin{pmatrix}
0.95 &  0.7\\
0.5 & 0.1
\end{pmatrix}.
$$
Since there are only two possible states, the state transition matrix for entering the sick state is given by calculating 1 minus each entry of $P$. In the code, the inputs will be as follows:


```python
state_party = ('healthy','sick')
action_party = ('relax','party')
prob_party = np.array([[[0.95, 0.7],[0.5, 0.1]],
                        [[0.05, 0.3],[0.5, 0.9]]])
reward_party = np.array([[7, 10], [0, 2]])
```

To run the value_it function, use the following line of code. The output will be an array, first giving the expected value of following the optimal policy, and then giving the optimal policy. 


```python
value_it(state_party,action_party, prob_party,reward_party,1000, 0.8)
```




    (array([35.71428571, 23.80952381]), ['party', 'relax'])



## A second example

To give a larger example of using the value iteration function, suppose we have a robot moving on a $3 \times 3$ grid, where the squares is the grid are labelled $a,b,c,d,e,f,g,h,i$, where $e$ is the middle square. If the robot hits the right edge of the grid, it takes a penalty of 20 points, and if it reaches square $e$, it gains a reward of $50$. The grid (with the expected rewards in brackets) for each space is given as follows:

| | | |
|---|---|---|
| a (0) | b (0) | c (-2) |
| d (0) | e (50)| f (-2) |
| g (0) | h (0) | i (-2) |
 

We suppose that if the robot is told to take an action (say for example, move up), there is a 70% chance it correctly follows the order, and a 10% chance of it moving down, left or right instead (that is, a 10% chance for each option). For example, if the robot is in square $b$, and the chosen action is to move up, the probability it moves to sqaure $a$ is 0.7 , and the probability it stays in square $b$ (because it moved left) is 0.1. The states, actions, transitions matrix and reward matrix can then be coded as follows:


```python
state_grid = ('a','b','c','d','e','f','g','h','i')
action_grid = ('up', 'down', 'right', 'left')

prob_grid = np.array([ [[0.8,0.2,0.2,0.8], [0.1,0.1,0.1,0.7], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0], [0,0,0,0],[0,0,0,0], [0,0,0,0]],
                      [[0.1,0.1,0.7,0.1],[0.7,0.1,0.1,0.1], [0.1,0.1,0.1,0.7], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
                      [[0,0,0,0], [0.1,0.1,0.7,0.1], [0.8,0.2,0.8,0.2], [0,0,0,0], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
                      [[0.1,0.7,0.1,0.1], [0,0,0,0], [0,0,0,0], [0.1,0.1,0.1,0.7], [0.1,0.1,0.1,0.7], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0]],
                      [[0,0,0,0], [0.1,0.7,0.1,0.1], [0,0,0,0], [0.1,0.1,0.7,0.1], [0,0,0,0], [0.1,0.1,0.1,0.7], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0]], 
                      [[0,0,0,0], [0.1,0.1,0.7,0.1], [0.8,0.2,0.8,0.2], [0,0,0,0], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
                      [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0.7,0.1,0.1,0.1], [0,0,0,0], [0,0,0,0], [0.2,0.8,0.2,0.8], [0.1,0.1,0.1,0.7], [0,0,0,0]], 
                      [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0.1,0.7,0.1,0.1], [0,0,0,0], [0.1,0.1,0.7,0.1], [0.1,0.7,0.1,0.1], [0.1,0.1,0.1,0.7]], 
                      [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0.1,0.7,0.1,0.1], [0,0,0,0], [0.1,0.1,0.7,0.1], [0.8,0.2,0.8,0.2]]
                      ]
    )

reward_grid = np.array([[0,0,0,0], [0,50,-2,0], [-2,-2,-20,0], [0,0,50,0], [0,0,-2,0], [-2,-2,-20,50], [0,0,0,0], [50,0,-2,0], [-2,-2,-20,0]])

```

We can then use the value_it function to find the optimal policy for which direction to tell the robot to move as follows, this time with a discount factor of 0.9:


```python
value_it(state_grid, action_grid, prob_grid, reward_grid, 1000, 0.9)
```




    (array([3.48226045e+154, 6.51401305e+154, 6.90686530e+154, 2.51545269e+154,
            3.14536516e+154, 5.93440873e+154, 1.39280874e+154, 1.63593013e+154,
            8.28813267e+153]),
     ['right', 'right', 'up', 'up', 'up', 'up', 'up', 'up', 'left'])


