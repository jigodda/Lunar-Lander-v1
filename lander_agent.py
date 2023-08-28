import gym
import numpy as np
import math
from numpy import savetxt, loadtxt

env = gym.make("LunarLander-v2")

#a numpy array which holds bins for all observations
def create_bins(num_bins_per_observation):
    xCord = np.linspace(-1.5, 1.5, num_bins_per_observation)
    yCord = np.linspace(-1.5, 1.5, num_bins_per_observation)
    xVelocity = np.linspace(-5.0, 5.0, num_bins_per_observation)
    yVelocity = np.linspace(-5.0, 5.0, num_bins_per_observation)
    angle = np.linspace(-math.pi, math.pi, num_bins_per_observation)
    angVelocity = np.linspace(-5.0, 5.0, num_bins_per_observation)
    #if the left leg is on the ground
    leftGround = np.linspace(0.0, 1.0, num_bins_per_observation)
    #if the right leg is on the ground
    rightGround = np.linspace(0.0, 1.0, num_bins_per_observation)
    bins = np.array([xCord, yCord, xVelocity, yVelocity, angle, angVelocity, leftGround, rightGround])
    
    return bins 

NUM_BINS = 5
BINS = create_bins(NUM_BINS)

#takes in observations and returns discretized version
def discretize_observation(observations, bins):
    
    binned_observations = []

    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)

    return tuple(binned_observations)

def nineToTwo(qTable):
        twoDTable = qTable.flatten()
        return twoDTable

def twoToNine(twoDTable):
    qTable = twoDTable.reshape(NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, 4)
    return qTable


qShape = (NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1, env.action_space.n)
qTable = np.zeros(qShape)
take2DTable = loadtxt('savedQTable.csv', delimiter=',')
qTable = twoToNine(take2DTable)

#returns random changes in behavior
def epsilon_greedy_action_selection(epsilon, qTable, discrete_state):

    if np.random.random() > epsilon:

        action = np.argmax(qTable[discrete_state])
    else:
        action = np.random.randint(0, env.action_space.n)

    return action
#returns the next Q Value
def computeNextQValue(oldQValue, reward, nextOptimalQValue):

    return oldQValue + ALPHA * (reward + GAMMA*nextOptimalQValue - oldQValue)

def reduceEpsilon(epsilon):
    epsilon -= EPSILON_REDUCE

    return epsilon

EPOCHS = 1

epsilon = 1
EPSILON_REDUCE = 0.0004
MIN_EPSILON = 0.1

ALPHA = 0.001
GAMMA = 0.999

success = 0
successBeforeEp = 0
successAfterEp = 0

for epoch in range(EPOCHS+1):
    initialState = env.reset()
    discretized_state = discretize_observation(initialState, BINS)
    done = False
    totalReward = 0.0

    while not done:

        action = epsilon_greedy_action_selection(epsilon, qTable, discretized_state)

        nextState, reward, done, info  = env.step(action)
        totalReward += reward

        nextStateDiscretized = discretize_observation(nextState, BINS)

        oldQValue = qTable[discretized_state + (action,)]
        nextOptimalQValue = np.max(qTable[nextStateDiscretized])

        nextQ = computeNextQValue(oldQValue, reward, nextOptimalQValue)
        qTable[discretized_state + (action,)] = nextQ

        discretized_state = nextStateDiscretized
    print("\nEpisode: ", epoch)    
    print("Reward: ", totalReward) 
    print("Epsilon: ", epsilon) 

    if totalReward >= 200:
        success += 1
        if epsilon > MIN_EPSILON:
            successBeforeEp += 1
        else:
            successAfterEp += 1

    print("Total Successes: ", success)
    #print results to output file when the simulation is over
    if epoch % EPOCHS == 0 and epoch != 0:
        twoDTable = nineToTwo(qTable)
        #savetxt('savedQTable.csv', twoDTable, delimiter=' ')

    if epsilon > MIN_EPSILON:
        epsilon = reduceEpsilon(epsilon)
print("Succeeded ", success, " times, or ", success/epoch, "% of the time")
print("Succeeded ", successBeforeEp, " times before epsilon reached min")
print("Succeeded ", successAfterEp, " times after epsilon reached min")
    


