import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from lander_agent import create_bins 
from lander_agent import discretize_observation
from lander_agent import computeNextQValue
from lander_agent import twoToNine
from numpy import loadtxt
from colorama import Fore, Back, Style

env = gym.make("LunarLander-v2")


NUM_BINS = 5
BINS = create_bins(NUM_BINS)

qShape = (NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, env.action_space.n)
qTable = np.zeros(qShape)
take2DTable = loadtxt('savedQTable.csv', delimiter=',')
qTable = twoToNine(take2DTable)


EPOCHS = 1000
ALPHA = 0.001
GAMMA = 0.999

superSuccess = 0
success = 0

for epoch in range(EPOCHS+1):
    initialState = env.reset()
    discretized_state = discretize_observation(initialState, BINS)
    done = False
    totalReward = 0.0

    while not done:

        action = action = np.argmax(qTable[discretized_state])

        nextState, reward, done, info = env.step(action)
        totalReward += reward

        nextStateDiscretized = discretize_observation(nextState, BINS)

        oldQValue = qTable[discretized_state + (action,)]
        nextOptimalQValue = np.max(qTable[nextStateDiscretized])

        

        discretized_state = nextStateDiscretized
        #env.render()

    print("\nEpisode: ", epoch)    
    print("Reward: ", totalReward) 

    
    
    if totalReward >= 90:
        success += 1
        if totalReward < 200:
            print(Fore.GREEN + 'Success!', Style.RESET_ALL)
        elif totalReward >= 200:
            superSuccess += 1
            print(Fore.RED + 'Super Success!', Style.RESET_ALL)
    print("Total successes/super successes: ", success, " ", superSuccess)

print("Succeeded (90+)", success, " times, or ", (success/epoch) * 100, "% of the time")
print("Super Succeeded (200+) ", superSuccess, " of those times, or ", (superSuccess/epoch) * 100, "% of the time")