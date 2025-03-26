import numpy as np

def value_it(States, Actions, Probs, Rewards, I = 1000, gamma = 0.8) :
    #States is the set of all states
    #Actions is the set of all actions
    #Probs is the state transition function, which will be an sxsxa matrix
    #Rewards is a reward function R(s,a) so is always 2 dimensional
    #I is number of iterations they want
    #gamma in the discount 
    s = len(S) #number of states
    a = len(A) #number of actions
    gamma = 0.8
    V = np.zeros((I+1, s))
    pi = [0]*s
    for k in range(1, (I+1)):
        arg_max = [0]*s 
        for i in range(0,s):
            action_vec = [0]*a
            for j in range(0,a):
                sum_vec = [0]*s
                for n in range(0,s):
                    sum_vec[n] = P[n,i,j]*V[k-1,n]
                action_vec[j] = R[i,j] + gamma*sum(sum_vec)
            V[k,i] = max(action_vec)
            arg_max[i] = np.argmax(action_vec)
            pi[i] = A[arg_max[i]]
    print(V[(k),])
    return (V[(k),], pi)
