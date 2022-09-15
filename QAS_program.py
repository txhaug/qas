#nt# -*- coding: utf-8 -*-
"""


@author: Tobias Haug, tobias.haug@u.nus.edu
Program for Quantum assisted simulator (QAS) for NISQ computers
Also includes Generalized Quantum assisted simulator (GQAS) for open systems
Quantum-assisted simulator, K Bharti, T Haug, Physical Review A 104 (4), 042418

Generalized quantum assisted simulator, T Haug, K Bharti, Quantum Science and Technology 7 (4), 045019

#to reproduce Fig. 3 of GQAS, set open_system=0 (closed), open_system=2 (for open), gamma=1, model=8, ini_type=2, ini_evolve_type=1

Prepares state on quantum computer, then measure it to simulate time evolution
First does IQAE to get initial state for evolution, then evolution with QAS or GQAS. 
Compute expectation value and fidelity with exact solution

"""



import qutip as qt

from functools import partial
import operator
from functools import reduce
import numpy as np
from matplotlib.ticker import AutoMinorLocator


import scipy

import time
#from helper_tools import *


import matplotlib.pyplot as plt

def plot1D(data,x,xlabelstring="",ylabelstring="",
           logx=False,logy=False,legend=[]):

    fig_size=(6,5)
    #self constructed from color brewer
    colormap=np.array([(56,108,176),(251,128,114),
                       (51,160,44),(253,191,111),(227,26,28),
                       (178,223,138),(166,206,227),(255,127,0),
                       (202,178,214),(106,61,154),(0,0,0)])/np.array([255.,255.,255.]) 
    
    elements=len(data)


    fsize=18
    fsizeLabel=fsize+12
    fsizeLegend=19
    
    plt.figure(figsize=fig_size)

    ax = plt.gca()
    

    plot1DLinestyle=flatten([["solid","dashed","dotted","dashdot"] for i in range(elements//4+1)])


    markerStyle=flatten([["o","X","v","^","s",">","<"] for i in range(elements//4+1)])

    markersize=6
    lwglobal=3
    tickerWidth=1.2
    minorLength=4
    majorLength=8
    linewidth=[lwglobal for k in range(elements)]

    dashdef=[]
    for i in range(elements):
        if(plot1DLinestyle[i]=="solid"):
            dashdef.append([1,1])
        elif(plot1DLinestyle[i]=="dotted"):
            dashdef.append([0.2,1.7])
            linewidth[i]*=1.7
        elif(plot1DLinestyle[i]=="dashed"):
            dashdef.append([3,2])
        elif(plot1DLinestyle[i]=="dashdot"):
            dashdef.append([5,2.5,1.5,2.5])

        
        else:
            dashdef.append([1,1])

            
    for i in range(elements):
        l,=plt.plot(x[i],data[i],color=colormap[i],linewidth=linewidth[i],
            linestyle=plot1DLinestyle[i],marker=markerStyle[i],ms=markersize,dash_capstyle = "round")
    
        
    if(logx==True):
        ax.set_xscale('log')

    if(logy==True):
        ax.set_yscale('log')
        

    plt.locator_params(axis = 'x',nbins=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.locator_params(axis = 'y',nbins=4)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
    plt.tick_params(axis ='both',which='both', width=tickerWidth)
    plt.tick_params(axis ='both',which='minor', length=minorLength)
    plt.tick_params(axis ='both', which='major', length=majorLength)



    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
                
    plt.xlabel(xlabelstring)
    plt.ylabel(ylabelstring)
    
    if(len(legend)>0):
        plt.legend(legend, fontsize=fsizeLegend,columnspacing=0.5)
            


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fsizeLabel)
    for item in ([]+ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
    

    

def prod(factors):
    return reduce(operator.mul, factors, 1)


def flatten(l):
    return [item for sublist in l for item in sublist]

def genFockOp(op,position,size,levels,opdim=0):
    opList=[qt.qeye(levels) for x in range(size-opdim)]
    opList[position]=op
    return qt.tensor(opList)




#multiplies paulis together
def multiply_paulis(curr_paulis,to_mult_paulis,curr_val_expansion=[],to_val_expansion=[]):
    new_paulis=[]
    new_val=[]
    for i in range(len(curr_paulis)):
        for j in range(len(to_mult_paulis)):
            add_pauli=np.zeros(len(curr_paulis[i]),dtype=int)

            for k in range(len(curr_paulis[i])):
                if(curr_paulis[i][k]==1 and to_mult_paulis[j][k]==2):
                    add_pauli[k]=3
                elif(curr_paulis[i][k]==2 and to_mult_paulis[j][k]==1):
                    add_pauli[k]=3
                else:
                    add_pauli[k]=np.abs(curr_paulis[i][k]-to_mult_paulis[j][k])

            new_paulis.append(add_pauli)
            if(len(curr_val_expansion)>0):
                new_val.append(curr_val_expansion[i]*to_val_expansion[j])

    #new_paulis=list(np.unique(new_paulis,axis=0))
    new_paulis,inverse_array=list(np.unique(new_paulis,axis=0,return_inverse=True))
    new_val=np.array(new_val)
    new_val_unique=np.zeros(len(new_paulis))
    
    #reconstruct weight values for each pauli, for paulis which occur multiple times are added up
    for i in range(len(new_val)):
        new_val_unique[inverse_array[i]]+=np.abs(new_val[i])
    

    return new_paulis,new_val_unique






def get_ini_state(ini_state_type):
    global anneal_time_opt
    #get initial state

    if(ini_state_type==0):#product state plust state
        initial_state=qt.tensor([qt.basis(levels,1)+qt.basis(levels,0) for i in range(n_qubits)])
        
        #initial_state=qt.tensor([qt.basis(levels,1)]+[qt.basis(levels,0) for i in range(L-1)]) #tjis was used for paper to compare against imag time evolution
        


    elif(ini_state_type==1):#all 0
        initial_state=qt.tensor([qt.basis(levels,0) for i in range(n_qubits)])
        
    elif(ini_state_type==2): #random state
    
        rand_angles=np.random.rand(depth,n_qubits)*2*np.pi
        rand_pauli=np.random.randint(1,4,[depth,n_qubits])

        entangling_layer=prod([opcsign[j] for j in range(n_qubits-1)])
        initial_state=qt.tensor([qt.basis(levels,0) for i in range(n_qubits)])
        initial_state=qt.tensor([qt.qip.operations.ry(np.pi/4) for i in range(n_qubits)])*initial_state

        for j in range(depth):

            rot_op=[]
            for k in range(n_qubits):
                angle=rand_angles[j][k]
                if(rand_pauli[j][k]==1):
                    rot_op.append(qt.qip.operations.rx(angle))
                elif(rand_pauli[j][k]==2):
                    rot_op.append(qt.qip.operations.ry(angle))
                elif(rand_pauli[j][k]==3):
                    rot_op.append(qt.qip.operations.rz(angle))
                    

            initial_state=qt.tensor(rot_op)*initial_state

            initial_state=entangling_layer*initial_state



    initial_state/=initial_state.norm()
    
    return initial_state


#get pauli strings for models
def get_Hamiltonian_string(L,model,J,h):
    Hstrings=[]
    Hvalues=[]
    Hoffset=0
    
    if(model==0):#ising
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(0.5*J)
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(0.5*h)
    
    elif(model==1):#heisenberg
    
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(h)
                
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=2
                paulistring[(i+1)%L]=2
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
        

            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                paulistring[(i+1)%L]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
            

            
    elif(model==8):#ising ladder
        if(J!=0):
            Lrung=L//2
            for i in range(Lrung-1):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[i+1]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(0.25*J)
            for i in range(Lrung-1):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i+Lrung]=3
                paulistring[i+1+Lrung]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(0.25*J)
            for i in range(Lrung):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[i+Lrung]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(0.25*J) 
                
                
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(h)
                
    elif(model==14):# transvesre ising with longitudonal field
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(h)     
        if(g!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(g)    
                

            
        
        
    HpauliFactor=np.zeros([len(Hstrings),L,4])
    for i in range(len(Hstrings)):
        pauliFactor=np.zeros([L,4])
        for j in range(L):
            if(Hstrings[i][j]==0):
                pauliFactor[j]=[1,0,0,0]
            elif(Hstrings[i][j]==1):
                pauliFactor[j]=[0,1,0,0]
            elif(Hstrings[i][j]==2):
                pauliFactor[j]=[0,0,1,0]
            elif(Hstrings[i][j]==3):
                pauliFactor[j]=[0,0,0,1]
        HpauliFactor[i]=pauliFactor
    return Hstrings,HpauliFactor,Hvalues,Hoffset


#make operator from pauli string
def get_pauli_op(pauli_string):
    pauli_circuit=opId
    for i in range(len(pauli_string)):
        if(pauli_string[i]!=0):
            if(pauli_string[i]==1):
                pauli_circuit=pauli_circuit*opX[i]
            elif(pauli_string[i]==2):
                pauli_circuit=pauli_circuit*opY[i]
            elif(pauli_string[i]==3):
                pauli_circuit=pauli_circuit*opZ[i]

    return pauli_circuit


#evolve beta in time
def evolveLindblad(t,y,evolveH,evolveFMatrix=[],evolveRMatrix=[],evolveRConjMatrix=[],imag=1):
    matrixLength=len(y)
    matrixShape=[int(np.sqrt(matrixLength)),int(np.sqrt(matrixLength))]
    beta=np.reshape(y,matrixShape)
    
    #apply to beta
    evolvedH=np.dot(imag*evolveH,beta)

    n_lindblad=len(evolveFMatrix)
    if(n_lindblad>0):
        evolvedF=sum([np.dot(evolveFMatrix[n],beta) for n in range(n_lindblad)])
        #evolveR=sum([np.dot(np.dot(evolveRMatrix[n],beta),evolveRConjMatrix[n]) for n in range(n_lindblad)])
        evolveR=sum([np.dot(np.dot(evolveRMatrix[n],beta),np.transpose(np.conjugate(evolveRMatrix[n]))) for n in range(n_lindblad)])
        

        res=-1j*(evolvedH-np.transpose(np.conjugate(evolvedH)))+evolveR-0.5*(evolvedF+np.transpose(np.conjugate(evolvedF)))

    else:
        res=-1j*(evolvedH-np.transpose(np.conjugate(evolvedH)))

            
    return np.reshape(res,matrixLength)


def numberToBase(n, b,n_qubits):
    if n == 0:
        return np.zeros(n_qubits,dtype=int)
    digits = np.zeros(n_qubits,dtype=int)
    counter=0
    while n:
        digits[counter]=int(n % b)
        n //= b
        counter+=1
    return digits[::-1]
                





starttime=time.time()


seed=1
np.random.seed(seed)

n_qubits=6#number of qubits

#state to use as ansatz for k-moment expansion 
ini_type=2 #initial state for ansatz state via K moment, 0: + product state, 1: zero product state, 2: random circuit

ini_evolve_type=1 ##state to use for actual time evolution, is generated via IQAE. See ini_type for type of states. Use same state as the one used for K-moment expansion by setting to -1

depth=6#layers of circuit

gamma=1 ##parameter for lindblad
open_system=0##0: closed system 1: lowering operator on every qubit, 2: lindblad raising operator
tfinal=6
n_timesteps=101


model=8 #model to be optimized: 0: transverse ising, 1: Heisenberg , 8: transverse ladder model, 14: transverse ising model with longitudinal field, 30: find optimal POVMs for state discrimination
h=1 #for models the h parameter
J=1 #for models the J parameter
g=0


hilbertspace=2**n_qubits
##range of expansion


max_expansion=2## how many orders of K-moment expansion should be performed
max_states=0 #maximal number of ansatz states, set to zero to get maximal possible number given max_expansion
##tune this to see how many ansatz states needed to get high fidelity for QAS


inv_cond=1e-10 #conditioning factor for inversion and QAE, increase if norm is diverging 




times=np.linspace(0,tfinal,num=n_timesteps)


qt_dims=[[2 for i in range(n_qubits)],[2 for i in range(n_qubits)]]

  
levels=2
#define operators
opZ=[genFockOp(qt.sigmaz(),i,n_qubits,levels) for i in range(n_qubits)]
opX=[genFockOp(qt.sigmax(),i,n_qubits,levels) for i in range(n_qubits)]
opY=[genFockOp(qt.sigmay(),i,n_qubits,levels) for i in range(n_qubits)]

opId=genFockOp(qt.qeye(levels),0,n_qubits,levels)
opLower=[genFockOp(qt.sigmap(),i,n_qubits,levels) for i in range(n_qubits)] #lowers from 1 to 0 (sigmap() is inversely defined...)
opRaise=[genFockOp(qt.sigmam(),i,n_qubits,levels) for i in range(n_qubits)] #raises from 0 to 1 (sigmam() is inversely defined...)


if(n_qubits>1):
    opcsign=[qt.qip.operations.csign(n_qubits,i,(i+1)%n_qubits) for i in range(n_qubits)]

##whether to use sparse solver to get ground state as reference
if(n_qubits>10):
    sparse_gs=True
else:
    sparse_gs=False
  
    

    
lindblad_strings=[]
if(open_system==1): ##lowering operator lindblad on every qubit
    lindblad=[]
    if(gamma!=0):
        lindblad= [np.sqrt(gamma)*opLower[i] for i in range(n_qubits)]
        for i in range(gamma):
            string=np.zeros(n_qubits,dtype=int)
            string[i]=1
            lindblad_strings.append(string)
            string=np.zeros(n_qubits,dtype=int)
            string[i]=2
            lindblad_strings.append(list(string))
            
elif(open_system==2): #raising operator on every qubit


    lindblad=[]
    if(gamma!=0):
        lindblad= [np.sqrt(gamma)*opRaise[i] for i in range(n_qubits)]
        for i in range(gamma):
            string=np.zeros(n_qubits,dtype=int)
            string[i]=1
            lindblad_strings.append(string)
            string=np.zeros(n_qubits,dtype=int)
            string[i]=2
            lindblad_strings.append(list(string))

if(open_system!=0):
    n_lindblad=len(lindblad)
    lindblad2=[]
    if(gamma!=0):
        lindblad2= [lindblad[i].dag()*lindblad[i] for i in range(n_lindblad)]




Hstrings,HpauliFactor,Hvalues,Hoffset=get_Hamiltonian_string(n_qubits,model,J,h)
H=0
for i in range(len(Hvalues)):
    H+=Hvalues[i]*get_pauli_op(Hstrings[i])
H+=Hoffset
target_op=H






initial_state=get_ini_state(ini_type) #get quantum state used as basis for cumulative K-moment expansion

if(ini_evolve_type<0):##get state that actually needs to be evolved
    initial_evolve_state=initial_state ##use same state as K-moment expansion
else:
    initial_evolve_state=get_ini_state(ini_evolve_type)  ##use different state, find via IQAE


opt=qt.Options()#options for solver taken from qutip

dt = np.diff(times)

#exact evolution with exact dynamics
if(open_system==0):
    res_exact=qt.mesolve(H,initial_evolve_state,times,[],[])
    exact_states=res_exact.states

else:
    res_exact=qt.mesolve(H,initial_evolve_state,times,lindblad,[])
    exact_states=res_exact.states 
    
        
    
    
#define state to do moment expansion with


##generate all possible pauli strings
n_paulis=2**n_qubits
all_pauli_list=np.zeros([n_paulis,n_qubits],dtype=int)
for k in range(n_paulis):
    all_pauli_list[k,:]=numberToBase(k,2,n_qubits)





##generate K-moment expansion that serves as linear combination basis of the subspace
base_expand_strings=[np.zeros(n_qubits,dtype=int)]


for k in range(max_expansion): ##do K_moment expansion up to max_expansion

    #do moment expansion with Hamiltonian terms.
    base_expand_strings+=list(multiply_paulis(base_expand_strings,Hstrings)[0])
    new_strings,string_index=list(np.unique(base_expand_strings,axis=0, return_index=True))
    sorted_index=np.sort(string_index)
    base_expand_strings=[base_expand_strings[k] for k in sorted_index]
    

    ####base_expand_strings=list(new_strings)
    

    if(max_states>0): ##select only a subset of states for the ansatz
        base_expand_strings=base_expand_strings[:max_states]





#get maximal subspace 
expand_strings=base_expand_strings
expand_states=[]
n_expand_states=len(base_expand_strings)
for i in range(n_expand_states):
    expand_states.append(get_pauli_op(base_expand_strings[i])*initial_state)

print("Number of ansatz states used",n_expand_states)
    
E_matrix=np.zeros([n_expand_states,n_expand_states],dtype=np.complex128)

D_matrix=np.zeros([n_expand_states,n_expand_states],dtype=np.complex128)

if(open_system!=0):
    R_matrix=np.zeros([n_lindblad,n_expand_states,n_expand_states],dtype=np.complex128) #matrix L_k term
    F_matrix=np.zeros([n_lindblad,n_expand_states,n_expand_states],dtype=np.complex128) #matrix L_k^\dag L_k term

    
for m in range(n_expand_states):
    for n in range(n_expand_states):
        E_matrix[m,n]=expand_states[m].overlap(expand_states[n])

        D_matrix[m,n]=(expand_states[m].overlap(target_op*expand_states[n]))
        if(open_system!=0):
            for k in range(n_lindblad):
                F_matrix[k,m,n]=expand_states[m].overlap(lindblad2[k]*expand_states[n])
            for k in range(n_lindblad):
                R_matrix[k,n,m]=expand_states[n].overlap(lindblad[k]*expand_states[m])






 
     

#adjust E_matrix to make it numerical stable for positive definite

#E_matrix can be not positive definite due to numerical issues. Diagonalize matrix, and set negative eigenvalues to positiv value
#this is only needed for QAE, not for QAS


e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
#    e_vals_adjusted=np.array(e_vals)
#    
#
#    for k in range(len(e_vals_adjusted)):
#        if(e_vals_adjusted[k]<epsilon):
#            e_vals_adjusted[k]=epsilon
#    E_matrix_corrected=np.dot(e_vecs,np.dot(np.diag(e_vals_adjusted),np.transpose(np.conjugate(e_vecs))))
#    
#    

##calculate inverse of E
##inv_cond determines threshold for SVD values: below inv_cond, value is set to zero. This is to avoid problems with small eigenvalues, which can blow up with inversion
##E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)



#choose initial alpha. Here, we choose initial_state as beginning state. In principle, any alpha can be chosen. One could also use QAE to find some state
ini_alpha_vec=np.zeros(n_expand_states,dtype=np.complex128)



if(ini_evolve_type>=0):
    #prepare initial state via QAE

    #get closest projection of initial evolution state via IQAE
    ##NOTE: one could here simply pick a Hamiltonian, i.e. to get the 0 state one would choose a hamiltonian of the form H=-\sum_i \sigma_i^z
    ini_projector=-initial_evolve_state*initial_evolve_state.dag()
    ini_matrix=np.zeros([n_expand_states,n_expand_states],dtype=np.complex128)
    for m in range(n_expand_states):
        for n in range(m,n_expand_states):
            ini_matrix[m,n]=expand_states[m].overlap(ini_projector*expand_states[n])
    for m in range(n_expand_states):
        for n in range(m+1,n_expand_states):
            ini_matrix[n,m]=np.conjugate(ini_matrix[m,n])
    


            
    #get e_matrix eigenvalues inverted, cutoff with inv_cond
    e_vals_inverted=np.array(e_vals)

    for k in range(len(e_vals_inverted)):
        if(e_vals_inverted[k]<inv_cond):
            e_vals_inverted[k]=0
        else:
            e_vals_inverted[k]=1/e_vals_inverted[k]
            
    #get e_matrix eigenvalues conditioned, such that small/negative eigenvalues are set to zero
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0

#            E_matrix_corrected=np.dot(e_vecs,np.dot(np.diag(e_vals_adjusted),np.transpose(np.conjugate(e_vecs))))
#            
  
    #diffEmatrix=np.sum(np.abs(E_matrix_corrected-E_matrix))
    #if(diffEmatrix>10**-13):
    #    print("WARN: Difference of E_matrix and corrected exceeded threshold, adjust epsilon to take care of positive definiteness",diffEmatrix)
    
    #if(n_samples>0):
    #    print("WARN: Replace actual E_matrix with one where eigenvalue have been cutoff with threshold",epsilon )
    #    E_matrix=E_matrix_corrected    
    
    #calculate ground state via generalized eigenvalue problem D\alpha=\lambda E\alpha
    #qae_energy,qae_vectors=scipy.linalg.eigh(ini_matrix,E_matrix_corrected)
    
    ###calculate eigenvalues for E^-1*H\alpha=\lambda \alpha. DOesnt seem to work for some resason though....
    ##E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)
    #qae_energy,qae_vectors=scipy.linalg.eigh(np.dot(E_inv,ini_matrix))


    #E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)

    #convert generalized eigenvalue problem with a regular eigenvalue problem using "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve D\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonlaize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*D*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business
    s_matrix=np.dot(e_vecs,np.diag(np.sqrt(e_vals_cond)))
    s_matrix_inv=np.dot(np.diag(np.sqrt(e_vals_inverted)),np.transpose(np.conjugate(e_vecs)))
    toeigmat=np.dot(s_matrix_inv,np.dot(ini_matrix,np.transpose(np.conjugate(s_matrix_inv))))

    qae_energy,qae_vectors=scipy.linalg.eigh(toeigmat)

    
    ini_alpha_vec=qae_vectors[:,0] 
    ini_alpha_vec=np.dot(np.transpose(np.conjugate(s_matrix_inv)),ini_alpha_vec) ##multiply with inv smatrix


    norm_ini_alpha=np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(ini_alpha_vec)),np.dot(E_matrix,ini_alpha_vec))))
    ini_alpha_vec=ini_alpha_vec/norm_ini_alpha ##normlaize states, in general not necessary i think
    
    ##reconstruct states corresponding to IQAE solution, we use this to compute fidelity
    ini_reconstructed_state=sum([expand_states[i]*ini_alpha_vec[i] for i in range(n_expand_states)])

        

    ini_norm=ini_reconstructed_state.norm()
    
        

    fidIni=np.abs(ini_reconstructed_state.overlap(initial_evolve_state))**2

    
    print("Done IQAE to find initial state for evolution")
    print("Initial state fidelity with exact initial state",fidIni,"alpha ini norm",norm_ini_alpha,"state reconstructed norm",ini_norm)

    
else:
    #fix evolution state as the initial ansatz state
    print("Pick same state as state prepared on quantum computer (for K-moment expansion) for evolution")
    ini_alpha_vec[0]=1




id_matrix=np.diag(np.ones(np.shape(E_matrix)))



E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)
evolveMatrix=np.dot(E_inv,D_matrix)


if(open_system==0):
    alpha_evolve=np.zeros([len(times),n_expand_states],dtype=np.complex128)


    ##compute D_matrix for evolution with Hamiltonian
    H_toeigmat=np.dot(s_matrix_inv,np.dot(D_matrix,np.transpose(np.conjugate(s_matrix_inv))))

    H_energy,H_vectors=scipy.linalg.eigh(H_toeigmat)
    
    ##does expansion into eigenbasis of D_matrix for evolution, turns out to be more stable
    alpha_eig_vectors=[np.dot(np.transpose(np.conjugate(s_matrix_inv)),H_vectors[:,i]) for i in range(n_expand_states)]
    
    ##computes coefficient with each eigenstate of D_matrix
    coefficients_eig=np.array([np.dot(np.transpose(np.conjugate(alpha_eig_vectors[i])),np.dot(E_matrix,ini_alpha_vec)) for i in range(n_expand_states)])




    ##evolution terms of alpha vector for QAS
    alpha_time=[sum([coefficients_eig[i]*alpha_eig_vectors[i]*np.exp(-1j*H_energy[i]*times[j]) for i in range(n_expand_states)]) for j in range(len(times))]

    ##computes actual quantum state from QAS vector.
    ##This is possible only for small qubit number.
    ##For larger systems, one can instead compute expectation values from alpha vector and appropiate measurmennts (follow how the D_matrix is computed, but now with operator of choice)
    evolved_QAE_state=[sum([expand_states[i]*alpha_time[j][i] for i in range(n_expand_states)]) for j in range(len(times))]

        
    ##compute norm to see all is good
    norm_evolve_eig=[evolved_QAE_state[j].norm() for j in range(len(times))]
    
    
    ##compute fidelity with exact solution
    fidelity_evolution=[np.abs(evolved_QAE_state[j].overlap(exact_states[j]))**2 for j in range(len(times))]

 


    
    
else:
    ##open system dynmaics with GQAS
    ini_beta=np.outer(ini_alpha_vec,np.conjugate(ini_alpha_vec)) #make sure this is positive semidefinite



    norm_ini_beta=np.real(np.trace(np.dot(ini_beta,E_matrix)))
        
        
    print("initial beta norm",norm_ini_beta)
    ini_beta_normed=ini_beta/norm_ini_beta

    ##F and R matrix from GQAS
    evolveFMatrix=[np.dot(E_inv,F_matrix[k]) for k in range(n_lindblad)]
    evolveRMatrix=[np.dot(E_inv,R_matrix[k]) for k in range(n_lindblad)]
    #evolveRConjMatrix=[np.dot(np.transpose(R_matrix[k]),E_inv) for k in range(n_lindblad)]
    #evolveRConjMatrix=[np.dot(np.conjugate(R_matrix[k]),E_inv) for k in range(n_lindblad)]
    #evolveRConjMatrix=[np.dot(np.transpose(np.conjugate(R_matrix[k])),E_inv) for k in range(n_lindblad)]
    
    beta_evolve=[ini_beta_normed]
    
    matrixShape=np.shape(beta_evolve[0])
    matrixLength=matrixShape[0]*matrixShape[1]
    
    ##evolve with zvode using equation of motions of GQAS
    evolveFunc=partial(evolveLindblad,evolveH=evolveMatrix,evolveFMatrix=evolveFMatrix,evolveRMatrix=evolveRMatrix)
    
    normalize_beta=True#normalize beta after every step, needed when number of ansatz states is low


    for t_idx, t in enumerate(times[:-1]):
        if(normalize_beta==True):
            r = scipy.integrate.ode(evolveFunc)
            r.set_integrator('zvode', method=opt.method, order=opt.order,
                             atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                             first_step=opt.first_step, min_step=opt.min_step,
                             max_step=opt.max_step)
            r.set_initial_value(np.reshape(beta_evolve[t_idx],matrixLength), times[t_idx])
        
        r.integrate(r.t + dt[t_idx])
        beta=np.reshape(r.y,matrixShape)
        if(normalize_beta==True):
            norm_beta=np.trace(np.dot(beta,E_matrix))
            beta=beta/norm_beta
            

            
        beta_evolve.append(beta)
    
    ##compute purity of state from beta
    purity_QAS=[np.real(np.trace(np.dot(np.dot(beta_evolve[j],E_matrix),np.dot(beta_evolve[j],E_matrix))))  for j in range(len(times))]

    dims=(expand_states[0]*expand_states[0].dag()).dims
    
    dense_array=[expand_states[i].data.toarray() for i in range(n_expand_states)]
    ##reconstruct actual density matrix from beta matrix
    costructed_states=[[np.dot(dense_array[i],np.transpose(np.conjugate(dense_array[k]))) for i in range(n_expand_states)] for k in range(n_expand_states)]
    
    #costructed_states=[[(expand_states[i]*expand_states[k].dag()).data.toarray() for i in range(n_expand_states)] for k in range(n_expand_states)]
    ##actual dm
    evolved_QAE_state=[sum([sum([beta_evolve[j][i][k]*costructed_states[k][i] for i in range(n_expand_states)]) for k in range(n_expand_states)]) for j in range(len(times))]
    
    evolved_QAE_state=[qt.Qobj(evolved_QAE_state[j],dims=dims) for j in range(len(times))]
    ##fidelity between evolved state and exact solution
    fidelity_evolution=[qt.fidelity(evolved_QAE_state[i],exact_states[i]) for i in range(len(times))]
    

        

       
#get various expectation values
expectZ=[np.real(qt.expect(opZ[k],evolved_QAE_state))  for k in range(n_qubits)] 

expectX=[np.real(qt.expect(opX[k],evolved_QAE_state))  for k in range(n_qubits)]
            


expectH=np.real(qt.expect(target_op,evolved_QAE_state))


expectZpure=[np.real(qt.expect(opZ[k],exact_states)) for k in range(n_qubits)]
    
expectXpure=[np.real(qt.expect(opX[k],exact_states))  for k in range(n_qubits)]
expectHpure=np.real(qt.expect(target_op,exact_states))


##plot fidelity
plot1D([fidelity_evolution],[times]*(2),xlabelstring="$t$",ylabelstring="$F$")



#this is same Ladder ising model as considered by "variational quantum simulation of general processes"
expectZZ=[]
expectZZpure=[]
if(model==8):
    Lrungs=n_qubits//2
    couplings=[[k,k+1] for k in range(Lrungs-1)]+[[k+Lrungs,k+1+Lrungs] for k in range(Lrungs-1)]+[[k,k+Lrungs] for k in range(Lrungs)]
    expectZZ=[sum([np.real(qt.expect(opZ[c[0]]*opZ[c[1]],evolved_QAE_state[j]))  for c in couplings])/len(couplings) for j in range(len(times))]
    expectZZpure=[sum([np.real(qt.expect(opZ[c[0]]*opZ[c[1]],exact_states[j]))  for c in couplings])/len(couplings) for j in range(len(times))]
    legendOrderPurity=["QAS"]+["exact"]
    plot1D([expectZZ]+[expectZZpure],[times]*(2),xlabelstring="$t$",ylabelstring="$\\langle Z_i Z_j\\rangle$",legend=legendOrderPurity)



print("Total time taken",time.time()-starttime)


