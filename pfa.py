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
import torch.optim
from rfutils import lazy_property, memoize

colon = slice(None)
newaxis = None
log2 = log(2)
INF = float('inf')
EOS = '!!</S>!!'

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
        """ As we are defining it, a PFA has three parameters:
        
        pi: The action policy, a probability distribution on actions A given goal states G and memory states M.
            This is represented as a GxMxA tensor, where G is the number of goal states, M is the number of memory states, and A is the number of possible actions.
            The last dimension of pi gives the probability p(A|G, M), so it should sum to 1. 
        mu: The memory policy, a probability distribution on memory states given actions and previous memory states.
            Represented as an AxMxM tensor, where A is the number of actions and M is the number of memory states.
        source: A probability distribution on goal states. Represented as a vector. Usually [1], indicating that we effectively aren't using goal states.

        """
        assert source.ndim == 1, source.ndim
        assert mu.ndim == 3, mu.ndim
        assert pi.ndim == 3, pi.ndim

        assert source.shape[0] == pi.shape[0]
        assert pi.shape[2] == mu.shape[0]
        assert pi.shape[1] == mu.shape[1] == mu.shape[2]
        
        self.pi = pi # matrix G x M x A -> Prob
        self.mu = mu # matrix A x M x M -> Prob
        self.source = source # vector G -> Prob

    @property
    def is_unifilar(self):
        # Because G influences A beyond M,
        # G has to be included in the definition of "state" for unifilarity.
        # So we enforce H[G] == 0 for unifilarity.
        # Conceivably, there are cases where H[G]>0 but we have unifilarity,
        # but these are strange and don't seem important now.
        def conditions():
            yield entropy(self.source) == 0

            # Because of the way we've defined the PFA,
            # we know H[M' | G, A, M] = H[M' | A, M].
            # Now check that that is equal to zero:
            yield conditional_entropy(self.mu_joint, self.mu) == 0
        return all(conditions())

    def marginalize_source(self):
        return PDFA(torch.Tensor([1]), torch.stack([self.state_emission_matrix]), self.mu)

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
        # WARNING: Does not properly take source into account! Only ok if entropy(source)==0.
        assert entropy(self.source) == 0
        
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
        h_last = self.entropy_rate_estimates[-1]
        return sum(h - h_last for h in entropy_rate_estimates)
    
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

class ConditionalPDFA(PFA):
    def __init__(self, *a, **k):
        super(ConditionalPDFA, self).__init__(*a, **k)
        assert self.is_conditionally_unifilar

    @lazy_property
    def is_conditionally_unifilar(self):
        G = self.source.shape[0]
        def conditions():
            for g in range(G):
                gvec = torch.Tensor(one_hot(g, G))
                yield PFA(gvec, self.pi, self.mu).is_unifilar
        return all(conditions())

class PDFA(PFA):
    """ PDFA differs from PFA in that mu is deterministic.
    Therefore, we can calculate entropy rate and conditional entropy rate in closed form. 
    """
    def __init__(self, *args, **kwds):
        super(PDFA, self).__init__(*args, **kwds)
        self.check_unifilar()

    def check_unifilar(self):
        if not self.is_unifilar:
            badness = conditional_entropy(self.mu_joint, self.mu) + entropy(self.source)
            print("Warning: PDFA object is not actually deterministic: Badness %s" % badness, file=sys.stderr)
    
    @lazy_property
    def entropy_rate(self):
        """ Formula from Shannon (1948). Only for PDFA. """
        self.check_unifilar()
        joint = self.stationary_M[:, newaxis] * self.state_emission_matrix
        return conditional_entropy(joint, self.state_emission_matrix)


    def excess_entropy_estimate(self, k):
        """ Lower bound estimate of excess entropy """
        entropy_rate_estimates = [self.entropy_rate_estimate(t) for t in range(k+1)]
        return sum(h - self.entropy_rate for h in entropy_rate_estimates)


def code_to_pdfa_with_delimiter(code):
    # Produce a PFA that generates a code.
    # The PFA is not guaranteed to be minimal: in fact it usually won't be.
    # It is deterministic conditional on the source symbols.
    # Code is a dictionary from goals to sequences of symbols.
    # e.g., a -> 0, b -> 10, c -> 110, d -> 111
    # PFA will generate 0#, 10#, 110#, 111#, etc.
    code = dict(code)
    
    # Will need canonically ordered lists of goals and codes:
    goals = sorted(list(code.keys()))
    seqs = [code[g] for g in goals]
    
    N = len(code) # number of source symbols
    
    vocab = [EOS] + list(frozenset.union(frozenset(), *code.values()))
    V = len(vocab)

    # Source is uniform over the goals:
    source = torch.ones(N) / N

    # Transitions go deterministically into the next state,
    # or into the first state if the sequence is over. That means
    # that the memory state generally has to fully encode the goal,
    # or that there has to be an end-of-sequence symbol.
    # Here we use the end-of-sequence symbol:
    num_states = max(len(v) for v in code.values()) + 1

    # In general, regardless of the emitted symbol, we always
    # transition into the next state; except if the emitted symbol
    # is EOS, then we go into the first state.
    # (i.e., I[A:M'|M] = 0)
    # Transition tensor mu has dimensions AxMxM
    # We start by generating a generic MxM that sends you into the next state.
    M_next = torch.cat([torch.zeros(num_states)[None, :], torch.eye(num_states)]).T[:num_states, :num_states]

    # We will also need the matrix that sends everything into the initial state:
    M_final = torch.zeros(num_states, num_states)
    M_final[:,0] = 1

    # Finally, mu uses M_final for EOS (the first vocab symbol),
    # and M_next for everything else:
    mu = torch.stack([M_final] + [M_next for _ in range(V - 1)])

    # Now we need to generate pi.
    # Pi is GxMxA, so we first iterate through goals:
    pi = [
        [
            one_hot(vocab.index(seq[i] if i<len(seq) else EOS), V)
            for i in range(num_states)
        ]
        for g, seq in zip(goals, seqs)
    ]
    pi = torch.Tensor(pi)

    return ConditionalPDFA(source, pi, mu)
            
def one_hot(k, n):
    y = [0 for _ in range(n)]
    y[k] = 1
    return y

def maxent(pdfa, num_epochs=5000, **kwds):
    """ Given a PDFA, return a new PDFA with the same transition function mu
    but with an emission distribution pi that maximizes entropy rate.

    The emission distribution pi is optimized by gradient descent by PyTorch;
    this function takes keyword parameters that will be passed to the
    optimized. 

    Note that the resulting emission distribution pi cannot have any 
    zero entries, but they may be very small.
    """
    # Find pi to maximize the entropy rate give source and mu
    # This is a convex problem so gradient descent will converge to the global optimum
    source = torch.Tensor(pdfa.source)
    pi = torch.Tensor(pdfa.pi)
    mu = torch.Tensor(pdfa.mu)
    pi_logit = torch.tensor(logit(epsilonify(pi)), requires_grad=True)
    opt = torch.optim.Adam(params=[pi_logit], **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        pi = torch.softmax(pi_logit, dim=-1)
        h = -PDFA(source, pi, mu).entropy_rate
        h.backward()
        opt.step()
        if i % 500 == 0:
            print((-h).item(), file=sys.stderr)
    return PDFA(source, pi, mu)
    
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

def epsilonify(x, eps=10**-10):
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
    
def no_ab_example():
    source = np.array([1])
    pi = np.array([  
        [[1/4,1/4,1/4, 1/4],
         [1/3,1/3,0,1/3]]             
    ])
    mu = np.array([
        [[1, 0], # symbol #
         [1, 0]],
        [[0, 1], # symbol a
         [0, 1]],
        [[1, 0], # symbol b
         [0, 0]],
        [[1, 0], # symbol c
         [1, 0]]
    ])

    no_ab = PDFA(source, pi, mu)
    return no_ab

def exists_ab_example():
    source = np.array([1])
    pi = np.array([  
        [[0, 1/3, 1/3, 1/3], # from state 0, emit a with prob. 1/2, or b with prob 1/2
         [0, 1/3, 1/3, 1/3], # from state 1, emit a with prob. 1/2, or b with prob 1/2
         [1/4, 1/4, 1/4, 1/4]], # from state 2, emit # with prob 1/2, a with prob 1/4, or b with prob 1/4
    ])
    mu = np.array([
        [[0, 0, 0], 
         [0, 0, 0], 
         [1, 0, 0]], # upon emitting #, from state 2, go into state 0
        [[0, 1, 0], 
         [0, 1, 0], 
         [0, 0, 1]], # upon emitting a, from state 0 or 1, go into state 1; from 2, go into 2
        [[1, 0, 0], 
         [0, 0, 1], 
         [0, 0, 1]], # upon emitting b, from state 0 go into state 0; from 1 go into 2; from 2 go into 2
        [[1, 0, 0], 
         [1, 0, 0], 
         [0, 0, 1]] # symbol c
    ])

    E_ab = PDFA(source, pi, mu)
    return E_ab

def exists_ab_canonical_example():
    source = np.array([1])
    pi = np.array([  
        [[0, 1/2, 1/2], # from state 0, emit a or b
         [0, 1/2, 1/2], # from state 1, emit a or b
         [0, 1/2, 1/2], # from state 2, emit a or b
         [1/2, 1/4, 1/4]], # from state 3, emit #, a, or b
    ])
    mu = np.array([
        [[0, 0, 0, 0], # upon emitting #, from state 0, go nowhere -- this state is impossible
        [0, 0, 0, 0], # impossible
        [0, 0, 0, 0], # impossible
        [1, 0, 0, 0]], # upon emitting #, from state 2, go into state 0
        [[0, 1, 0, 0],  # upon emitting a, from state 0, go into state 1
        [0, 1, 0, 0], # upon emitting a, from 1, go into 1
        [0, 0, 0, 1], # upon emitting a, from 2, go into 3
        [0, 0, 0, 1]], # upon emitting a, from 3, go into 3
        [[0, 0, 1, 0], # upon emitting b, from 0, go into 2
        [0, 0, 0, 1], # upon emitting b, from 1, go into 3
        [0, 0, 1, 0], # upon emitting b, from 2, go into 2
        [0, 0, 0, 1]], # upon emitting b, from 3, go into 3
    ])

    exists_ab_canonical = PDFA(source, pi, mu)
    return exists_ab_canonical

def one_ab_example():
    source = np.array([1])
    pi = np.array([
        [[0, 1/3, 1/3, 1/3],
         [0, 1/3, 1/3, 1/3],
         [1/4, 1/4, 1/4, 1/4],
         [1/3, 1/3, 0, 1/3]]
    ])
    mu = np.array([
        [[0, 0, 0, 0], # on symbol #
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0]],
        [[0, 1, 0, 0], # on symbol a
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 1]],
        [[1, 0, 0, 0], # on symbol b
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 0], # on symbol c
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]   
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

def strictly_piecewise_example():
    source = np.array([1])
    pi = np.array([
        [[1/4, 1/4, 1/4, 1/4],
         [1/3, 1/3, 0, 1/3]],
    ])

    mu = np.array([
        [[1, 0], # on symbol #
         [1, 0]],
        [[0, 1], # on symbol a
         [0, 1]],
        [[1, 0], # on symbol b
         [0, 0]],
        [[1, 0], # on symbol c
         [0, 1]]
    ])

    strictly_piecewise = PDFA(source, pi, mu)
    return strictly_piecewise

def piecewise_testable_example():
    source = np.array([1])
    pi = np.array([
        [[0, 1/3, 1/3, 1/3],
         [0, 1/3, 1/3, 1/3],
         [1/4, 1/4, 1/4, 1/4]]
    ])

    mu = np.array([
        [[0, 0, 0], # on symbol #
         [0, 0, 0],
         [1, 0, 0]],
        [[0, 1, 0], # on symbol a
         [0, 1, 0],
         [0, 1, 0]],
        [[1, 0, 0], # on symbol b
         [0, 0, 1],
         [0, 0, 1]],
        [[1, 0, 0], # on symbol c
         [0, 1, 0],
         [0, 0, 1]]
    ])
    
    piecewise_testable = PDFA(source, pi, mu)
    return piecewise_testable


def star_free_example():
    source = np.array([1])
    pi = np.array([
        [[1/3, 1/3, 0, 1/3],
         [0, 1/3, 1/3, 1/3],
         [1/4, 1/4, 1/4, 1/4]]
    ])

    mu = np.array([
        [[1, 0, 0], # on symbol #
         [0, 0, 0],
         [1, 0, 0]],
        [[0, 1, 0], # on symbol a
         [0, 1, 0],
         [0, 0, 1]],
        [[0, 0, 0], # on symbol b
         [0, 0, 1],
         [0, 0, 1]],
        [[1, 0, 0], # on symbol c
         [0, 1, 0],
         [0, 0, 1]]
    ])

    star_free = PDFA(source, pi, mu)
    return star_free

def strictly_piecewise_factored_a_example():
    source = np.array([1])
    pi = np.array([
        [[1/3, 1/3, 1/3],
         [1/2, 1/2, 0]]
    ])

    mu = np.array([
        [[1, 0], # on symbol #
         [1, 0]],
        [[0, 1], # on symbol a
         [0, 1]],
        [[1, 0], # on symbol b
         [0, 0]]
    ])

    strictly_piecewise_factored = PDFA(source, pi, mu)
    return strictly_piecewise_factored

def strictly_piecewise_factored_b_example():
    source = np.array([1])
    pi = np.array([
        [[1/3, 1/3, 1/3],
         [1/3, 1/3, 1/3]]
    ])

    mu = np.array([
        [[1, 0], # on symbol #
         [1, 0]],
        [[1, 0], # on symbol a
         [0, 1]],
        [[0, 1], # on symbol b
         [0, 1]]
    ])

    strictly_piecewise_factored = PDFA(source, pi, mu)
    return strictly_piecewise_factored

def strictly_piecewise_factored_product_example():
    source = np.array([1])
    pi = np.array([
        [[1/3, 1/3, 1/3],
        [1/2, 1/2, 0],
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]]
    ])

    mu = np.array([
        [[1, 0, 0, 0], # on symbol #
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0]],
        [[0, 1, 0, 0], # on symbol a
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 1]],
        [[0, 0, 1, 0], # on symbol b
         [0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
    ])

    strictly_piecewise_factored = PDFA(source, pi, mu)
    return strictly_piecewise_factored

def strictly_local_three_example():
    source = np.array([1])
    pi = np.array([
        [[1/4, 1/4, 1/4, 1/4],
         [1/4, 1/4, 1/4, 1/4],
         [1/3, 1/3, 1/3, 0]]
    ])

    mu = np.array([
        [[1, 0, 0], # on symbol #
         [1, 0, 0],
         [1, 0, 0]],
        [[0, 1, 0], # on symbol a
         [0, 1, 0],
         [0, 1, 0]],
        [[1, 0, 0], # on symbol b
         [0, 0, 1],
         [1, 0, 0]],
        [[1, 0, 0], # on symbol c
         [1, 0, 0],
         [0, 0, 0]]
    ])
    strictly_local_three = PDFA(source, pi, mu)
    return strictly_local_three

def sl2_nt_example():
    source = np.array([1])
    pi = np.array([
        [[1/4, 1/4, 1/4, 1/4],
         [1/4, 1/4, 1/4, 1/4],
         [1/4, 1/4, 1/4, 1/4],
         [1/3, 0, 1/3, 1/3]]
    ])
    mu = np.array([
        [[1, 0, 0, 0], # on symbol #
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0]],
        [[0, 1, 0, 0], # on symbol T
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 1, 0], # on symbol D
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0]],
        [[0, 0, 0, 1], # on symbol N
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1]]   
    ])

    sl2_nt = PDFA(source, pi, mu)
    return sl2_nt


if __name__ == '__main__':
    import nose
    nose.runmodule()
