""" For tractability, refactor everything as a PFA. 
Should be able to differentiate the PFA and find the best one..."""

# Problem: To make a differentiable space of all PDFAs, need to include PFAs as well.
# But for PFAs, we have no closed-form expression for entropy rate.
# That's troublesome, but regularizing against I_mu[0] will make the PFA approximately deterministic.

# Problem: Most PFAs we move through might be expressible through other, more minimal machines...
# that means that the statistical complexity measure is not correct.
# Question: Is it possible to derive the mixed-state representation through a kind of spectral decomposition?

# Proposed objective function:
# max I[G:A_t | A_{<t}] - A*I_pi - B*I_mu
# For B[0] large, we will tend towards PDFAs.
# For B large, we will minimize statistical complexity.
# Problem: the "information rate" is hard to compute, because
# introducing the source random variable G means that the PFA becomes
# nondeterministic at each word boundary, a
import sys
from math import log, exp

import numpy as np
import torch
from rfutils import lazy_property, memoize

colon = slice(None)
newaxis = None
log2 = log(2)
INF = float('inf')

def stationary(M):
    if isinstance(M, torch.Tensor):
        return stationary_torch(M)
    else:
        return stationary_numpy(M)

def stationary_numpy(M):
    """ Find the stationary distribution in a way that is differentiable """
    n = M.shape[-1]
    k = M.ndim
    A = transpose(M, -1, -2) - np.eye(n)
    colons = (colon,)*(k - 2)
    A[colons + (-1, colon,)] = 1 # make the last row 1
    against = np.zeros(n)
    against[-1] = 1
    result = np.linalg.solve(A, against[(newaxis,)*(A.ndim-2)+(colon,)])
    return result

def stationary_torch(M):
    n = M.shape[-1]
    k = M.ndim
    A = transpose(M, -1, -2) - torch.eye(n)
    colons = (colon,)*(k-2)
    A[colons+(-1,colon,)] = 1
    against = torch.zeros(n)
    against[-1] = 1
    return torch.solve(against[:, newaxis], A).solution.T[-1].T # craziness
    
def transpose(x, one, two):
    if isinstance(x, torch.Tensor):
        return x.transpose(one, two)
    else: 
        indices = list(range(x.ndim))
        indices[one], indices[two] = indices[two], indices[one]
        return x.transpose(indices)

def entropy(ps):
    if isinstance(ps, torch.Tensor):
        P = torch
    else:
        P = np
    logs = P.log(ps)
    logs[P.isinf(logs)] = 0
    return -(ps * logs).sum() / log2

def conditional(joint_ps):
    num_conditioning_axes = joint_ps.ndim - 1
    denominator = joint_ps.sum(axis=-1) # sum out the last thing
    try:
        denominator[denominator == 0] = INF # so that division by zero leads to zero, not nan
    except TypeError:
        pass
    result = joint_ps / denominator[(colon,)*num_conditioning_axes + (newaxis,)]
    return result

def conditional_entropy(joint_ps, cond_ps):
    if isinstance(joint_ps, np.ndarray):
        P = np
        assert isinstance(cond_ps, np.ndarray)
    else:
        P = torch    
    logs = P.log(cond_ps)
    logs[P.isinf(logs)] = 0
    return -(joint_ps * logs).sum() / log2

def mutual_information(joint_ps, cond_ps, marginal_ps):
    return entropy(marginal_ps) - conditional_entropy(joint_ps, cond_ps)

class PFA(object):
    def __init__(self, source, pi, mu):
        self.pi = pi # matrix G x M x A -> Prob
        self.mu = mu # matrix A x M x M -> Prob
        self.source = source # vector G -> Prob

    @property
    def is_unifilar(self):
        # Because G influences A beyond M,
        # G has to be included in the definition of "state" for unifilarity.
        # So:
        return conditional_entropy(self.mu_joint, self.mu) + entropy(self.source) == 0

    def generate(self):
        num_g = self.source.shape[-1]
        gs = range(num_g)
        if isinstance(self.source, torch.Tensor):
            source = self.source.detach().numpy()
        else:
            source = self.source
        while True:
            g = np.random.choice(gs, p=source)
            yield from self.generate_from(g)
            
    def generate_from(self, g):
        num_symbols = self.pi.shape[-1]
        num_states = self.mu.shape[-1]

        symbols = range(num_symbols)
        states = range(num_states)

        if isinstance(self.pi, torch.Tensor):
            pi = self.pi.detach().numpy()
        else:
            pi = self.pi


        if isinstance(self.mu, torch.Tensor):
            mu = self.mu.detach().numpy()
        else:
            mu = self.mu

        init = 0
        m = init
        
        while True:
            a = np.random.choice(symbols, p=pi[g,m,:])
            yield (m, a)
            m = np.random.choice(states, p=mu[a,m,:])
            if m == init:
                break

    @lazy_property
    def state_emission_matrix(self):
        """ M x A vector giving p(a_t | m_t) """
        return (self.source[:, newaxis, newaxis] * self.pi).sum(axis=0)

    @lazy_property
    def stationary_M(self):
        """ M vector giving p(m_t) """
        return stationary(self.state_transition_matrix)

    @lazy_property
    def stationary_A(self):
        """ A vector giving p(a_t) """
        M = self.stationary_M[:, newaxis]
        return (M * self.state_emission_matrix).sum(axis=0)

    @lazy_property
    def state_transition_matrix(self):
        """ M x M matrix giving p(m_{t+1} | m_t) """
        E = self.state_emission_matrix.T[:, :, newaxis]
        return (self.mu * E).sum(axis=0)

    @lazy_property
    def statistical_complexity(self):
        return entropy(self.stationary_M)

    @lazy_property
    def action_transition_matrix(self):
        """ AxA matrix giving p(a_{t+1}|a_t) """
        return self.iterated_action_transition_matrix(1)

    @memoize
    def iterated_joint_matrix(self, k):
        """ AxAx...xA matrix of dimension k+1, giving p(a_{t+1}|m_{t+1}, a_t, m_t, ..., a_{t-k+1}) """
        initial = self.stationary_M
        integration = transpose(self.mu, 0, 1)#self.mu.transpose((1,0,2))
        emission = self.state_emission_matrix

        # We will need to build a gigantic (2k+1)-dimensional array
        # This is what the logic looks like typed out for k=2:
        #big_joint = (
        #    self.stationary_M[:, newaxis, newaxis, newaxis] *        # M_t x   1 x       1 x       1
        #    self.state_emission_matrix[:, :, newaxis, newaxis] *     # M_t x A_t x       1 x       1
        #    self.mu.transpose((1, 0, 2))[:, :, :, newaxis] *         # M_t x A_t x M_{t+1} x       1
        #    self.state_emission_matrix[newaxis, newaxis, :, :]       #   1 x   1 x M_{t+1} x A_{t+1}
        #)        
        emission_axes = [
            (newaxis,)*(2*t) + (colon,)*2 + (newaxis,)*(2*(k-t))
            for t in range(k+1)
        ]

        integration_axes = [
            (newaxis,)*(2*(t-1)) + (colon,)*(t>0)*2 + (colon,) + (newaxis,)*(2*(k-t)+1)
            for t in range(k+1)            
        ]

        giant_joint = initial[integration_axes[0]] * emission[emission_axes[0]] # start with m_t, a_t
        for t in range(k):
            giant_joint = giant_joint * integration[integration_axes[t+1]] # generate m_{t+1}
            giant_joint = giant_joint * emission[emission_axes[t+1]] # generate a_{t+1}
        return giant_joint

    @memoize
    def iterated_joint_action_matrix(self, k):
        giant_joint = self.iterated_joint_matrix(k)
        num_axes = 2*(k+1)
        m_axes = range(0, num_axes, 2) # all even numbers between 0 and 2*(k+1) non-inclusive
        return giant_joint.sum(axis=tuple(m_axes)) # sum out all the memory states

    @memoize
    def iterated_action_transition_matrix(self, k):
        return conditional(self.iterated_joint_action_matrix(k))

    def entropy_rate_estimate(self, k):
        return conditional_entropy(
            self.iterated_joint_action_matrix(k),
            self.iterated_action_transition_matrix(k)
        )

    def excess_entropy_estimate(self, k):
        """ Lower bound estimate of excess entropy """
        entropy_rate_estimates = [self.entropy_rate_estimate(t) for t in range(k+1)]
        return sum(entropy_rate_estimates[-1] - self.entropy_rate)
    

    def crypticity_estimate(self, k):
        """ Upper bound estimate of crypticity """
        S = self.statistical_complexity
        EE = self.excess_entropy_estimate(k)
        return S - EE

    @lazy_property
    def state_integration_matrix(self):
        """ AxM matrix giving p(m_{t+1}|a_t) """
        return self.mu_joint.sum(axis=1) / self.mu_joint.sum(axis=(1,2))[:, newaxis]

    @lazy_property
    def conditional_state_transition_matrix(self):
        """ G x M x M matrix giving p(m_{t+1} | g, m_t) """
        # NOT COVERED BY TESTS
        #return (self.pi[:,:,newaxis,:] * self.mu.transpose((1,2,0))[newaxis,:,:,:]).sum(axis=-1)
        mu = transpose(transpose(self.mu, 0, 2), 0, 1) # like np.transpose(x, (1,2,0))
        return (self.pi[:,:,newaxis,:] * mu[newaxis,:,:,:]).sum(axis=-1)
    
    @lazy_property
    def conditional_stationary_M(self):
        """ G x M matrix giving p(m|g) """
        return stationary(self.conditional_state_transition_matrix)

    @lazy_property
    def conditional_stationary_A(self):
        """ G x A matrix giving p(a|g) """
        return (self.conditional_stationary_M[:,:,newaxis] * self.pi).sum(axis=1)

    @lazy_property
    def pi_joint(self):
        """ G x M x A matrix giving p(g, m_t, a_t) """
        return (
            self.source[:, newaxis, newaxis] *
            self.conditional_stationary_M[:, :, newaxis] *
            self.pi
        )

    @lazy_property
    def mu_joint(self):
        """ A x M x M matrix giving p(a_t, m_t, m_{t+1}) """
        return (
            self.stationary_M[newaxis, :, newaxis] *
            self.state_emission_matrix.T[:, :, newaxis] *
            self.mu
        )

    @lazy_property
    def state_given_action(self):
        """ AxM matrix giving p(m_t | a_t) """
        return (self.pi_joint.sum(axis=0)/self.stationary_A[newaxis,:]).T

    @lazy_property
    def state_uncertainty(self):
        """ H[M_t | A_t], upper bound on crypticity """
        p_a_m = self.pi_joint.sum(axis=0).T # get p(a_t, m_t)
        return conditional_entropy(p_a_m, self.state_given_action)

    @lazy_property
    def excess_entropy_lower_bound(self):
        return self.statistical_complexity - self.state_uncertainty

    @lazy_property
    def I_pi(self):
        # First H[A_t | M_t, G]
        H_A_GM = conditional_entropy(self.pi_joint, self.pi)

        # Next I[A_t:G]
        # need p(a_t, g) = \sum_m p(a_t, m, g)
        p_a_g = self.pi_joint.sum(axis=1)
        I_G_A = mutual_information(p_a_g, self.conditional_stationary_A, self.stationary_A)

        # Next I[A_t : M_t]
        p_a_m = self.pi_joint.sum(axis=0)
        I_M_A = mutual_information(p_a_m, self.state_emission_matrix, self.stationary_A)

        # Next -I[A_t : M_t : G]
        Syn_A_M_G = (entropy(self.stationary_A) - H_A_GM) - I_G_A - I_M_A

        return [H_A_GM, I_G_A, I_M_A, Syn_A_M_G]

    @lazy_property
    def I_mu(self):
        # First H[M_{t+1} | M_t, A_t] --- zero iff it's a PDFA
        H_M_MA = conditional_entropy(self.mu_joint, self.mu)

        # Next I[M_{t+1}:A_t]
        p_m_a = self.mu_joint.sum(axis=1) # marginalize over M_t
        I_M_A = mutual_information(p_m_a, self.state_integration_matrix, self.stationary_M)

        # Next I[M_{t+1} : M_t]
        p_m_m = self.mu_joint.sum(axis=0) # marginalize over A_t
        I_M_M = mutual_information(p_m_m, self.state_transition_matrix, self.stationary_M)

        # Next -I[M_{t+1} : M_t : A_t]
        Syn_M_M_A = (entropy(self.stationary_M) - H_M_MA) - I_M_A - I_M_M

        return [H_M_MA, I_M_A, I_M_M, Syn_M_M_A]

class PDFA(PFA):
    """ PDFA differs from PFA in that mu is deterministic.
    Therefore, we can calculate entropy rate and conditional entropy rate in closed form. 
    """
    def __init__(self, *args, **kwds):
        super(PDFA, self).__init__(*args, **kwds)

    def check_unifilar(self):
        if not self.is_unifilar:
            badness = conditional_entropy(self.mu_joint, self.mu) + entropy(self.source)
            print("Warning: Calculating entropy rate as if PFA were deterministic, but it is not: Badness %s" % badness, file=sys.stderr)
    
    @lazy_property
    def entropy_rate(self):
        """ Formula from Shannon (1948). Only for PDFA. """
        self.check_unifilar()
        joint = self.stationary_M[:, newaxis] * self.state_emission_matrix
        return conditional_entropy(joint, self.state_emission_matrix)

    @lazy_property
    def conditional_entropy_rate(self):
        """ Entropy rate conditional on G. 
        This formula only works if I[G : A_t : A_{<t} | M] = 0 
        """
        joint = self.source[:, newaxis, newaxis] * self.conditional_stationary_M[:, :, newaxis] * self.pi
        return conditional_entropy(joint, self.pi)

    @lazy_property
    def information_rate(self):
        return self.entropy_rate - self.conditional_entropy_rate

    def excess_entropy_estimate(self, k):
        """ Lower bound estimate of excess entropy """
        entropy_rate_estimates = [self.entropy_rate_estimate(t) for t in range(k+1)]
        return sum(entropy_rate_estimates - self.entropy_rate)
    
def test_xor_example():
    """ 
    two meanings. p(g1) = .25, p(g2) = .75 
    three symbols, 0, 1, and #.
    four memory states: epsilon, 0, 1, and done.
    for meaning g1, produce either 00# or 11#, w.p. 1/2
    for meaning g2, produce either 01# or 10#, w.p. 1/2
    """
    epsilon = .00001
    
    source = np.array([.25, .75])
    pi = np.array([
        [[.5, .5, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],  # for goal g1 
        [[.5, .5, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],  # for goal g2
    ])

    mu = np.array([
        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]],   # for symbol 0
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]],   # for symbol 1
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]    # for symbol #
    ])

    pfa = PFA(source, pi, mu)
    assert np.all(pfa.state_emission_matrix == np.array(
       [[ 0.5 ,  0.5 ,  0.  ],  # from state 0, emit 0 w.p. 1/2, or 1 w.p. 1/2
        [ 0.25,  0.75,  0.  ],  # from state 1, emit 0 w.p. 1/4, or 1 w.p. 3/4
        [ 0.75,  0.25,  0.  ],  # from state 2, emit 0 w.p. 3/4, or 1 w.p. 1/4
        [ 0.  ,  0.  ,  1.  ]], # from state 3, emit # certainly
    ))

    assert np.all(pfa.state_transition_matrix == np.array(
        [[ 0. ,  0.5,  0.5,  0. ], # from state 0, go into state 1/2 |1> + 1/2 |1>
        [ 0. ,  0. ,  0. ,  1. ],  # from state 1, definitely go into state 3
        [ 0. ,  0. ,  0. ,  1. ],  # from state 2, definitely go into state 3
        [ 1. ,  0. ,  0. ,  0. ]]  # from state 3, definitely go into state 0
    ))

    assert np.all(pfa.state_integration_matrix - np.array(
       [[ 0. ,  0.5,  0. ,  0.5], # on action 0, go into memory state 1/2 |1> + 1/2 |3>
        [ 0. ,  0. ,  0.5,  0.5], # on action 1, go into memory state 1/2 |2> + 1/2 |3>
        [ 1. ,  0. ,  0. ,  0. ]] # on action #, go into memory state 0 w.p. 1
    ) < epsilon)


    assert np.abs(np.sum(pfa.I_pi) - entropy(pfa.stationary_A)) < epsilon
    assert np.abs(np.sum(pfa.I_mu) - entropy(pfa.stationary_M)) < epsilon

    return pfa

def logit(x):
    return torch.log(x/(1-x))

def epsilonify(x, eps=.00001):
    interval = 1 - 2*eps
    return (x * interval) + eps

def xor_torch():
    epsilon = .00001
    
    source = torch.tensor([.25, .75], requires_grad=False)
    # The underlying tensor is real-valued
    # No probabilities zero or one allowed
    pi = torch.tensor(logit(torch.tensor([
        [[.5, .5, epsilon],
         [1-epsilon, epsilon, epsilon],
         [epsilon, 1-epsilon, epsilon],
         [epsilon, epsilon, 1-epsilon]],  # for goal g1 
        [[.5, .5, epsilon],
         [epsilon, 1-epsilon, epsilon],
         [1-epsilon, epsilon, epsilon],
         [epsilon, epsilon, 1-epsilon]],  # for goal g2
    ], requires_grad=False)), requires_grad=True)

    mu = torch.tensor(logit(epsilonify(torch.tensor([
        [[0., 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]],   # for symbol 0
        [[0, 0, 1, 0],  [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]],   # for symbol 1
        [[1, 0, 0, 0],  [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],   # for symbol #
    ], requires_grad=False))), requires_grad=True)

    return source, pi, mu
    
def test_star_ab_example():
    source = np.array([1])
    # two memory states: q0 and q1
    # three symbols: #, a, and b
    # The language is *ab, so we generate things like #bbbbaaa#aaa#baaaaaaa##b#
    p_halt = .25
    pi = np.array([ 
        [[p_halt, (1-p_halt)/2, (1-p_halt)/2], # from state 0, we generate #, a, or b
         [p_halt, 1-p_halt, 0]] # from state 1, we generate # or a, never b
    ])
    mu = np.array([
        [[1, 0], [1, 0]],  # on action #, from state 0, we go into state 0, and from state 1, we go into state 0
        [[0, 1], [0, 1]],  # on action a, from state 0, we go into state 1, and from state 1, we go into state 1
        [[1, 0], [0, 0]], # on action b, from state 0, we go into state 0, and from state 1, undefined behavior
    ])

    ab = PDFA(source, pi, mu)

    epsilon = .000001

    assert np.abs(np.sum(ab.I_pi) - entropy(ab.stationary_A)) < epsilon
    assert np.abs(np.sum(ab.I_mu) - entropy(ab.stationary_M)) < epsilon
    assert np.abs(entropy(ab.stationary_M) - 0.970950594455) < epsilon
    assert np.abs(ab.excess_entropy_estimate(2) - 0.241446071166) < epsilon

    return ab

def test_pdfa_entropy_rate_simple():
    mu = np.array([[[1]], [[1]]])
    source = np.array([1])
    epsilon = .000001        
    probs = np.linspace(.01, .09)        
    for prob in probs:
        pi = np.array([
            [[prob, 1-prob]],
        ])
        machine = PDFA(source, pi, mu)
        reference_entropy = entropy(pi[0,0,:])
        assert np.abs(machine.entropy_rate - reference_entropy) < epsilon

def pdfa_entropy_rate_weird():
    # Note: The PDFA here is not the epsilon-machine for this language,
    # so asymptotic synchronization to the PDFA states is not guaranteed!
    # This code is not decodable for a maximum-entropy source,
    # but it would be if we add delimiter symbols (sync words).
    # Word boundaries = soft sync words.
    pi = np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ])
    mu = np.array([
        [[0, 1], [1, 0]],
        [[0, 1], [1, 0]],
    ])

    epsilon = .000001        
    probs = np.linspace(.01, .09)        
    for prob in probs:
        source = np.array([prob, 1 - prob])
        machine = PDFA(source, pi, mu)

        assert np.abs(machine.entropy_rate - entropy(source)) < epsilon
        assert np.abs(machine.conditional_entropy_rate - 0) < epsilon    
    
def no_ab_canonical_example():
    source = np.array([1])
    pi = np.array([  
        [[1/4, 3/8, 3/8],
         [1/4, 3/4, 0],
         [1/4, 0, 3/4]]             
    ])
    mu = np.array([
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [0, 0, 1]],
    ])

    # Unspecified:
    # mu = np.array([

    # ])
    no_ab_canonical = PDFA(source, pi, mu)
    return no_ab_canonical
    

def exists_ab_example():
    source = np.array([1])
    pi = np.array([  
        [[0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [1/2, 1/4, 1/4]],             
    ])
    mu = np.array([
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
    ])

    E_ab = PDFA(source, pi, mu)
    return E_ab

def exists_ab_canonical_example():
    source = np.array([1])
    pi = np.array([  
        [[0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [1/2, 1/4, 1/4]],             
    ])
    mu = np.array([
        [[0, 0, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [1, 0, 0, 0]],
        [[0, 1, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 1]],
        [[0, 0, 1, 0], 
        [0, 0, 0, 1], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]],
    ])

    exists_ab_canonical = PDFA(source, pi, mu)
    return exists_ab_canonical

def one_ab_example():
    source = np.array([1])
    pi = np.array([
        [[0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [1/2, 1/4, 1/4],
         [1/2, 1/2, 0],
         [1/3, 1/3, 1/3]]
    ])
    mu = np.array([
        [[0, 0, 0, 0, 0], # on symbol #
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0], # on symbol a
         [0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]],
        [[1, 0, 0, 0, 0], # on symbol b
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1]]    
    ])

    one_ab = PDFA(source, pi, mu)
    return one_ab


def one_ab_canonical_example():
    source = np.array([1])
    pi = np.array([
        [[0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [0, 1/2, 1/2],
         [1/2, 1/4, 1/4],
         [1/2, 1/2, 0],
         [1/3, 1/3, 1/3]]
    ])
    mu = np.array([
        [[0, 0, 0, 0, 0, 0], # on symbol #
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0], # on symbol a
         [0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0]],
        [[0, 1, 0, 0, 0, 0], # on symbol b
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]    
    ])

    one_ab_canonical = PDFA(source, pi, mu)
    return one_ab_canonical

def star_free_example():
    source = np.array([1])
    pi = np.array([
        [[0, 1, 0],
         [1/2, 1/4, 1/4]],
    ])

    mu = np.array([
        [[0, 0], # on symbol #
         [1, 0]],
        [[0, 1], # on symbol a
         [0, 1]],
        [[0, 0], # on symbol b
         [0, 1]]
    ])

    star_free = PDFA(source, pi, mu)
    return star_free


if __name__ == '__main__':
    import nose
    nose.runmodule()
