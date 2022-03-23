import numpy as np
import random as random
    
def select_random_without_replacement(population,number):
    return random.sample(list(enumerate(population)), number)

def estimate_fitness(selected, population, Z, A):
    
    P1_against_population=[population.count(0),population.count(1)]
    P2_against_population=[population.count(0),population.count(1)]
    
    if(selected[0][1]==0):
        P1_against_population[0]-=1
    else:
        P1_against_population[1]-=1
        
    if(selected[1][1]==0):
        P2_against_population[0]-=1
    else:
        P2_against_population[1]-=1
            
    ResultA=(((P1_against_population[0])*A[selected[0][1]][0])+((P1_against_population[1])*A[selected[0][1]][1]))/float(Z-1)
    ResultB=(((P2_against_population[0])*A[selected[1][1]][0])+((P2_against_population[1])*A[selected[1][1]][1]))/float(Z-1)
    
    return ResultA,ResultB

def prob_imitation(beta,fitness):
    return 1./(1. + np.exp(beta*(fitness[0]-fitness[1])))
    
def moran_step(current_state, beta, mu, Z, A):
    strategies=[0,1]
    selected = select_random_without_replacement(current_state, 2)
    fitness = estimate_fitness(selected, current_state, Z, A)
    
    if np.random.rand() < mu:
        current_state[selected[0][0]] = np.random.choice(strategies,size=1)[0]
    elif np.random.rand() < prob_imitation(beta,fitness):
        current_state[selected[0][0]] = current_state[selected[1][0]]
        
    return current_state
    
def estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, A):
    counter=np.zeros(51)
    for n in range(nb_runs):
        distribution = np.random.randint(Z)
        base_population=[]
        
        for i in range(distribution):
            base_population.append(0)
        for j in range(Z-distribution):
            base_population.append(1)
            
        transit=[]
        transit.append(base_population)
        
        for i in range(transitory):
            next_step=moran_step(transit[i],beta,mu,Z,A)
            transit.append(next_step)
        
        histo=[]
        histo.append(transit[transitory])
        for j in range(nb_generations):
            next_step=moran_step(histo[j],beta,mu,Z,A)
            histo.append(next_step)
            state=next_step.count(1)
            counter[state]+=1        

    counter /= float((nb_runs*nb_generations))
        
    return counter
    
def main():

    beta = 10
    mu = 1e-3
    Z=50
    
    transitory=10**3
    nb_generations=10**5
    nb_runs=10
    
    R=100
    A=np.array([[0,3],[-1,2*R]])
    A=A/R
    
    #Cheater will be 0
    #Copycat will be 1  
    
    return estimate_stationary_distribution(nb_runs,transitory,nb_generations,beta,mu,Z,A)
    
main()