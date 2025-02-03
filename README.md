# Quantum assisted simulator and Generalized quantum assisted simulator


Program for Quantum assisted simulator (QAS) for NISQ computers.
Also includes Generalized Quantum assisted simulator (GQAS) for open systems.

Quantum-assisted simulator, K Bharti, T Haug, Physical Review A 104 (4), 042418
Generalized quantum assisted simulator, T Haug, K Bharti, Quantum Science and Technology 7 (4), 045019

To reproduce Fig. 3 of "Generalized quantum assisted simulator", set open_system=0 (closed), open_system=2 (for open), gamma=1, model=8, ini_type=2, ini_evolve_type=1

Prepares state on quantum computer, then measures state in various bases.
Then follows only efficient post-processing of collected data:
First does IQAE to get initial state for evolution, then time evolves it with QAS or GQAS. 
Program then computes expectation values and fidelity with exact solution.
Supports various Hamiltonians and Lindbladians.

In contrast to Variational quantum algorithms, no parameterized circuit is needed, as well as no feedback loop.
The optimization program is efficient, and no barren plateau/getting stuck in local minimas is possible.


QAS has been demonstrated in experiment on IBM quantum computer in
Noisy intermediate scale quantum simulation of time dependent Hamiltonians, JWZ Lau, K Bharti, T Haug, LC Kwek, arXiv:2101.07677


IQAE part to get initial state is in more detailed explain in (for more code see https://github.com/txhaug/nisq-sdp)
Iterative quantum-assisted eigensolver, K Bharti, T Haug, Physical Review A 104 (5), L050401
or
Noisy intermediate-scale quantum algorithm for semidefinite programming, K Bharti, T Haug, V Vedral, LC Kwek, Physical Review A 105 (5), 052445
(this also features additional algorithms to find ground state of Hamiltonians, excited states, symmetry constrained problems and state discrimination problems)

NOTE: Requires older version of qutip, namely <=4.7.5, install via "pip install qutip==4.7.5"

@author: Tobias Haug, tobias.haug@u.nus.edu
