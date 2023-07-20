import numpy as np
#tranistions matrix
#0 fold, 1 check, 2 call, 3 bet , 4 raise
#prob to not hit anything=0,668, prob to hit pair=0,315, prob to hit set=0,017
#terminal states= shodown or fold
#q0-q4 = preflop states, q5-q19 = after flop state 
#q0-q4= 5 different cards,q5-q9=high card, q10-q14=pair, q15-q19=set
#q20 = terminal state
#(transition probability, next state, reawrd, is terminal)

class PolicyIterationAgent:
    def __init__(self, ptype):
        self.name = 'Policy Iteration Agent'
        if((ptype == 'threshold_tight') or (ptype == 'random')):
            self.P = P_vs_tight
        else:
            self.P = P_vs_loose
        
        self.pi = None


    def policy_evaluation(self, pi, P, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
        prev_V = np.zeros(len(P)) # use as "cost-to-go", i.e. for V(s')
        while True:
            V = np.zeros(len(P)) # current value function to be learnerd
            for s in range(len(P)):  # do for every state
                for prob, next_state, reward, done in P[s][pi(s)]:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < epsilon: #check if the new V estimate is close enough to the previous one;
                break # if yes, finish loop
            prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
        return V

    def policy_improvement(self, V, P, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64) #create a Q value array
        for s in range(len(P)):        # for every state in the environment/model
            for a in range(len(P[s])):  # and for every action in that state
                for prob, next_state, reward, done in P[s][a]:  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve)
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
        # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
        # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself

        return new_pi

    # policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

    def policy_iteration(self, P,gamma = 1.0, epsilon = 1e-10):

        action = []
        random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state
        pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]   
        t = 0
        while True:
            old_pi = {s: pi(s) for s in range(len(P))}  #keep the old policy to compare with new
            V = self.policy_evaluation(pi,P,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
            pi = self.policy_improvement(V,P,gamma)          #get a better policy using the value function of the previous one just calculated
            t += 1
            if old_pi == {s:pi(s) for s in range(len(P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        #print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        return V,pi


#tranistions matrix
#0 fold, 1 check, 2 call, 3 bet , 4 raise
#prob to not hit anything=0,668, prob to hit pair=0,315, prob to hit set=0,017
#terminal states= shodown or fold
#q0-q4 = preflop states, q5-q19 = after flop state 
#q0-q4= 5 different cards,q5-q9=high card, q10-q14=pair, q15-q19=set
#q20 = terminal state
#(transition probability, next state, reawrd, is terminal)
P_vs_loose = {
    0: {
        0: [(1, 20,20, True)
        ],
        1: [(0.668, 5, 20, False),
            (0.315, 10, 20, False),
            (0.017, 15, 20, False)
        ],
        2: [(0.668, 5, -10, False),
            (0.315, 10, -10, False),
            (0.017, 15, -10, False)
        ],
        3: [(0.668, 5, -200, False),
            (0.315, 10, -200, False),
            (0.017, 15, -200, False)
        ],
        4: [(0.668, 5, -200, False),
            (0.315, 10, -200, False),
            (0.017, 15, -200, False)
        ]
    },
    1: {
        0: [(1, 20, 20, True)
        ],
        1: [(0.668, 6, 20, False),
            (0.315, 11, 20, False),
            (0.017, 16, 20, False)
        ],
        2: [(0.668, 6, -20, False),
            (0.315, 11, -20, False),
            (0.017, 16, -20, False)
        ],
        3: [(0.668, 6, -200, False),
            (0.315, 11, -200, False),
            (0.017, 16, -200, False)
        ],
        4: [(0.668, 6, -200, False),
            (0.315, 11, -200, False),
            (0.017, 16, -200, False)
        ]
    },
    2: {
        0: [(1, 20, -20, True)
        ],
        1: [(0.668, 7, 20, False),
            (0.315, 12, 20, False),
            (0.017, 17, 20, False)
        ],
        2: [(0.668, 7, -20, False),
            (0.315, 12, -20, False),
            (0.017, 17, -20, False)
        ],
        3: [(0.668, 7, -200, False),
            (0.315, 12, -200, False),
            (0.017, 17, -200, False)
        ],
        4: [(0.668, 7, -200, False),
            (0.315, 12, -200, False),
            (0.017, 17, -200, False)
        ]
    },
    3: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 8, -200, False),
            (0.315, 13, -200, False),
            (0.017, 18, -200, False)
        ],
        2: [(0.668, 8, -200, False),
            (0.315, 13, -200, False),
            (0.017, 18, -200, False)
        ],
        3: [(0.668, 8, 200, False),
            (0.315, 13, 200, False),
            (0.017, 18, 200, False)
        ],
        4: [(0.668, 8, 200, False),
            (0.315, 13, 200, False),
            (0.017, 18, 200, False)
        ]
    },
    4: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 9, -200, False),
            (0.315, 14, -200, False),
            (0.017, 19, -200, False)
        ],
        2: [(0.668, 9, -200, False),
            (0.315, 14, -200, False),
            (0.017, 19, -200, False)
        ],
        3: [(0.668, 9, 200, False),
            (0.315, 14, 200, False),
            (0.017, 19, 200, False)
        ],
        4: [(0.668, 9, 200, False),
            (0.315, 14, 200, False),
            (0.017, 19, 200, False)
        ]
    },
    5: {
        0: [(1, 20, 20, True)],
        1: [(1, 20, 20, True)],
        2: [(1, 20, -20, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    6: {
        0: [(1, 20, 20, True)],
        1: [(1, 20, 20, True)],
        2: [(1, 20, -20, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    7: {
        0: [(1, 20, 0, True)],
        1: [(1, 20, 20, True)],
        2: [(1, 20, -20, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    8: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, 10, True)],
        2: [(1, 20, 20, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    9: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, 10, True)],
        2: [(1, 20, 20, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    10: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    11: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    12: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    13: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    14: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    15: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    16: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    17: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    18: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    19: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    20: {
        0: [(1, 20, 0, True)],
        1: [(1, 20, 0, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 0, True)],
        4: [(1, 20, 0, True)]
    }
}

P_vs_tight = {
    0: {
        0: [(1, 20,-200, True)
        ],
        1: [(0.668, 5, -200, False),
            (0.315, 10, -200, False),
            (0.017, 15, -200, False)
        ],
        2: [(0.668, 5, 200, False),
            (0.315, 10, 200, False),
            (0.017, 15, 200, False)
        ],
        3: [(0.668, 5, -200, False),
            (0.315, 10, -200, False),
            (0.017, 15, -200, False)
        ],
        4: [(0.668, 5, -200, False),
            (0.315, 10, -200, False),
            (0.017, 15, -200, False)
        ]
    },
    1: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 6, -200, False),
            (0.315, 11, -200, False),
            (0.017, 16, -200, False)
        ],
        2: [(0.668, 6, 200, False),
            (0.315, 11, 200, False),
            (0.017, 16, 200, False)
        ],
        3: [(0.668, 6, -200, False),
            (0.315, 11, -200, False),
            (0.017, 16, -200, False)
        ],
        4: [(0.668, 6, -200, False),
            (0.315, 11, -200, False),
            (0.017, 16, -200, False)
        ]
    },
    2: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 7, -200, False),
            (0.315, 12, -200, False),
            (0.017, 17, -200, False)
        ],
        2: [(0.668, 7, 200, False),
            (0.315, 12, 200, False),
            (0.017, 17, 200, False)
        ],
        3: [(0.668, 7, -200, False),
            (0.315, 12, -200, False),
            (0.017, 17, -200, False)
        ],
        4: [(0.668, 7, -200, False),
            (0.315, 12, -200, False),
            (0.017, 17, -200, False)
        ]
    },
    3: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 8, -200, False),
            (0.315, 13, -200, False),
            (0.017, 18, -200, False)
        ],
        2: [(0.668, 8, 200, False),
            (0.315, 13, 200, False),
            (0.017, 18, 200, False)
        ],
        3: [(0.668, 8, -200, False),
            (0.315, 13, -200, False),
            (0.017, 18, -200, False)
        ],
        4: [(0.668, 8, -200, False),
            (0.315, 13, -200, False),
            (0.017, 18, -200, False)
        ]
    },
    4: {
        0: [(1, 20, -200, True)
        ],
        1: [(0.668, 9, -200, False),
            (0.315, 14, -200, False),
            (0.017, 19, -200, False)
        ],
        2: [(0.668, 9, -200, False),
            (0.315, 14, -200, False),
            (0.017, 19, -200, False)
        ],
        3: [(0.668, 9, 200, False),
            (0.315, 14, 200, False),
            (0.017, 19, 200, False)
        ],
        4: [(0.668, 9, 200, False),
            (0.315, 14, 200, False),
            (0.017, 19, 200, False)
        ]
    },
    5: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    6: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    7: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    8: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    9: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    10: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    11: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    12: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 200, True)],
        3: [(1, 20, -200, True)],
        4: [(1, 20, -200, True)]
    },
    13: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    14: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    15: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    16: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    17: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    18: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    19: {
        0: [(1, 20, -200, True)],
        1: [(1, 20, -200, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 200, True)],
        4: [(1, 20, 200, True)]
    },
    20: {
        0: [(1, 20, 0, True)],
        1: [(1, 20, 0, True)],
        2: [(1, 20, 0, True)],
        3: [(1, 20, 0, True)],
        4: [(1, 20, 0, True)]
    }
}