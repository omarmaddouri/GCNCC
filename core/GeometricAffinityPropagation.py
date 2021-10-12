"""Affinity Propagation clustering algorithm."""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause

import scipy.sparse as sp
import networkx as nx
import sys
import os
from collections import defaultdict
from itertools import chain
from sklearn.metrics import pairwise_distances

import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import as_float_array, check_array, check_random_state
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin
import networkx.algorithms.community as nx_comm
import random

def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


@_deprecate_positional_args
def affinity_propagation(S, *, preference=None, convergence_iter=15,
                         max_iter=200, damping=0.5, copy=True, verbose=False,
                         return_n_iter=False, return_modularity = True, random_state='warn',
                         adjacency=None, randomized_network=False, node_similarity_mode='shortest_path',
                         node_similarity_threshold=1, save_mask=False,
                         saving_path=None, enable_label_smoothing=True, smoothing_depth=1, dataset=None):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=0
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------

    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it returns an empty array as
    ``cluster_center_indices`` and ``-1`` as label for each training sample.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy).astype(np.float32)
    n_samples = S.shape[0]
    #===========================================================================
    # Geometric AP: Omar Maddouri
    # Compute the mask based on the different metrics
    # [Start]: Geometric affinity propagation
    #===========================================================================
    smoothing_adjacency = adjacency.copy()
    np.fill_diagonal(smoothing_adjacency,1)
    
    if node_similarity_mode=='shortest_path':
        if not save_mask or not os.path.isfile("{}{}/mask_sp_{}.npy".format(saving_path, dataset, node_similarity_threshold)):
            filter = np.zeros(S.shape, dtype=np.int)
            G = nx.from_numpy_matrix(adjacency)
            if not isinstance(node_similarity_threshold, int):
                sys.exit("node_similarity_threshold for the mode shortest_path should be an integer")
            elif (node_similarity_threshold < 0):
                sys.exit("Invalid integer for communication hops")
            elif(node_similarity_threshold == 0):
                filter = np.ones(S.shape, dtype=np.int)
            elif(node_similarity_threshold == 1):
                filter = adjacency.astype(np.int)
            else:
                filter = adjacency.astype(np.int)
                for i in range(2, node_similarity_threshold+1):
                    filter += np.linalg.matrix_power(adjacency, i).astype(np.int)
            
            np.fill_diagonal(filter, 1)
            mask = (filter != 0)
            if(save_mask):
                np.save("{}{}/mask_sp_{}".format(saving_path, dataset, node_similarity_threshold), mask)
        else:
            mask = np.load("{}{}/mask_sp_{}.npy".format(saving_path, dataset, node_similarity_threshold), allow_pickle=True)
        
    elif node_similarity_mode=='jaccard_coefficient':
        if not save_mask or not os.path.isfile("{}{}/mask_jaccard_{}.npy".format(saving_path, dataset, node_similarity_threshold)):
            filter = pairwise_distances(adjacency, metric='jaccard')
            np.fill_diagonal(filter, 0)
            mask = (filter <= node_similarity_threshold)
            if(save_mask):
                np.save("{}{}/mask_jaccard_{}".format(saving_path, dataset, node_similarity_threshold), mask)
        else:
            mask = np.load("{}{}/mask_jaccard_{}.npy".format(saving_path, dataset, node_similarity_threshold), allow_pickle=True)
            
    elif node_similarity_mode=='cosine_similarity':
        if not save_mask or not os.path.isfile("{}{}/mask_cosine_{}.npy".format(saving_path, dataset, node_similarity_threshold)):
            filter = pairwise_distances(adjacency, metric='cosine')
            np.fill_diagonal(filter, 0)
            mask = (filter <= node_similarity_threshold)
            if(save_mask):
                np.save("{}{}/mask_cosine_{}".format(saving_path, dataset, node_similarity_threshold), mask)
        else:
            mask = np.load("{}{}/mask_cosine_{}.npy".format(saving_path, dataset, node_similarity_threshold), allow_pickle=True)
    
    #For testing purposes, we may randomize the network information and check the result
    if randomized_network:
        # Build ground truth graphs
        tmp_G_mask = nx.from_numpy_matrix(mask)
        tmp_G_adj = nx.from_numpy_matrix(smoothing_adjacency)
        # Create a random mappings
        random_mapping_mask = dict(zip(tmp_G_mask.nodes(), sorted(tmp_G_mask.nodes(), key=lambda k: random.random())))
        random_mapping_adj= dict(zip(tmp_G_adj.nodes(), sorted(tmp_G_adj.nodes(), key=lambda k: random.random())))
        # Build a new graphs from the random mappings
        random_G_mask = nx.relabel_nodes(tmp_G_mask, random_mapping_mask)
        random_G_adj = nx.relabel_nodes(tmp_G_adj, random_mapping_adj)
        # Generate the new random network information (nodelist should be the ground truth sequence)
        mask = nx.to_numpy_matrix(random_G_mask, nodelist=tmp_G_mask.nodes(), dtype=np.bool)
        smoothing_adjacency = nx.to_numpy_matrix(random_G_adj, nodelist=tmp_G_adj.nodes(), dtype=np.int)
        
        #=======================================================================
        # # Permute columns randomly
        # mask = np.random.permutation(mask.T).T
        # smoothing_adjacency = np.random.permutation(smoothing_adjacency.T).T
        # # Permute rows randomly
        # mask = np.random.permutation(mask)
        # smoothing_adjacency = np.random.permutation(smoothing_adjacency)
        # # Make sure self loops are conserved
        # np.fill_diagonal(mask, True)
        # np.fill_diagonal(smoothing_adjacency,1)
        #=======================================================================
        
    #===========================================================================
    # Geometric AP: Omar Maddouri
    # Compute the mask based on the different metrics
    # [End]: Geometric affinity propagation
    #===========================================================================
    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    preference = np.array(preference)

    if (n_samples == 1 or
            _equal_similarities_and_preferences(S, preference)):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn("All samples have mutually equal similarities. "
                      "Returning arbitrary cluster center(s).")
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return ((np.arange(n_samples), np.arange(n_samples), 0)
                    if return_n_iter
                    else (np.arange(n_samples), np.arange(n_samples)))
        else:
            return ((np.array([0]), np.array([0] * n_samples), 0)
                    if return_n_iter
                    else (np.array([0]), np.array([0] * n_samples)))

    if random_state == 'warn':
        warnings.warn(("'random_state' has been introduced in 0.23. "
                       "It will be set to None starting from 0.25 which "
                       "means that results will differ at every function "
                       "call. Set 'random_state' to None to silence this "
                       "warning, or to 0 to keep the behavior of versions "
                       "<0.23."),
                      FutureWarning)
        random_state = 0
    random_state = check_random_state(random_state)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples), dtype=np.float32)
    R = np.zeros((n_samples, n_samples), dtype=np.float32)  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Remove degeneracies
    S += ((np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) *
          random_state.randn(n_samples, n_samples).astype(np.float32))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))
    ind = np.arange(n_samples)
    
    #===========================================================================
    # Geometric AP: Omar Maddouri
    # Declare variables used to check for convergence after first perturbation
    # [Start]: Geometric affinity propagation
    #===========================================================================
    starting_config = np.zeros((n_samples, convergence_iter))
    check_convergence = False
    #===========================================================================
    # Geometric AP: Omar Maddouri
    # Declare variables used to check for convergence after first perturbation
    # [End]: Geometric affinity propagation
    #===========================================================================

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Compute the penalty and update the availability message
        # [Start]: Geometric affinity propagation
        #===========================================================================
        tmp -= np.sum(tmp, axis=0)
        penalty = np.where(mask, 0, tmp)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        tmp -= penalty
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Compute the penalty and update the availability message
        # [End]: Geometric affinity propagation
        #===========================================================================
        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)
        
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Print log over iterations
        # Check for convergence after first perturbation
        # [Start]: Geometric affinity propagation
        #===========================================================================
        if verbose:
            print("Message passing: iteration ({}/{})".format(it+1, max_iter), "--- {} potential exemplars identified".format(K))
        # Save the initial configuration if it holds for convergence_iter in order to skip it
        # Allow at least one configuration change after initial configuration to decide about convergence
        if it == (convergence_iter-1):
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)):
                starting_config = e.copy()
                
        if ( ((not check_convergence) and (it >= convergence_iter) and (e!=starting_config).any()) ) :
            check_convergence = True
        
        if ((not check_convergence) and (K > 0) and (it+1 == max_iter)):
            check_convergence = True
        # Check for convergence only if the initial configuration has changed at least one time
        if ( check_convergence and (it >= convergence_iter) ):
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Print log over iterations
        # Check for convergence after first perturbation
        # [End]: Geometric affinity propagation
        #===========================================================================
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0 and not never_converged:
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Assign data points to the closest exemplar among the neighbors
        # If no proximal exemplars, assign data points to the closest exemplar within the network
        # [Start]: Geometric affinity propagation
        #===========================================================================
        #Set similarity to non neighbor clusters to - infinity
        NB1 = np.where(mask[:,I], S[:, I], -np.inf)
        for row_idx in range(NB1.shape[0]):
            if not any(np.isfinite(NB1[row_idx,:])):
                NB1[row_idx,:] = S[row_idx, I]
        c = np.argmax(NB1, axis=1)
        
        smoothing_convergence = 0
        smoothing_c = c.copy()
        while enable_label_smoothing:
            for sl_idx in range(n_samples):
                neighbors_idx = np.where(smoothing_adjacency[sl_idx,:])
                neighbors_idx = list(chain(*neighbors_idx))
                smoothing_c[sl_idx] = np.argmax(np.bincount(c[neighbors_idx]))
            smoothing_convergence = smoothing_convergence + 1
            if ((smoothing_c ==  c).all()) or (smoothing_convergence>smoothing_depth):
                break
            else:
                c = smoothing_c.copy()
                
        c[I] = np.arange(K)  # Identify clusters
        
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))#Within the same cluster, select the data point that maximizes the similarity as exemplar
            I[k] = ii[j]
            
        #Redefine neighbor clusters because I has changed
        NB3 = np.where(mask[:,I], S[:, I], -np.inf)
        for row_idx in range(NB3.shape[0]):
            if not any(np.isfinite(NB3[row_idx,:])):
                NB3[row_idx,:] = S[row_idx, I]
        c = np.argmax(NB3, axis=1)
         
        smoothing_convergence = 0
        smoothing_c = c.copy()
        while enable_label_smoothing:
            for sl_idx in range(n_samples):
                neighbors_idx = np.where(smoothing_adjacency[sl_idx,:])
                neighbors_idx = list(chain(*neighbors_idx))
                smoothing_c[sl_idx] = np.argmax(np.bincount(c[neighbors_idx]))
            smoothing_convergence = smoothing_convergence + 1
            if ((smoothing_c ==  c).all()) or (smoothing_convergence>smoothing_depth):
                break
            else:
                c = smoothing_c.copy()
                
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Assign data points to the closest exemplar among the neighbors
        # If no proximal exemplars, assign data points to the closest exemplar within the network
        # [End]: Geometric affinity propagation
        #===========================================================================
        
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Compute Newman modularity
        # [Start]: Geometric affinity propagation
        #===========================================================================
        G = nx.from_numpy_matrix(adjacency)
        clusters = defaultdict(list)
        Partition_sets = []
        for i in range(len(labels)):
            clusters["Exemplar_{}".format(cluster_centers_indices[labels[i]])].append(i)
        for key, val in clusters.items():
            Partition_sets.append(set(val))
        modularity = nx_comm.modularity(G, Partition_sets)
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # Compute Newman modularity
        # [End]: Geometric affinity propagation
        #===========================================================================
        
        
    else:
        warnings.warn("Affinity propagation did not converge, this model "
                      "will not have any cluster centers.", ConvergenceWarning)
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # [Start]: Return center indices and labels even in non-convergence
        #===========================================================================
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
        #===========================================================================
        # Geometric AP: Omar Maddouri
        # [End]: Return center indices and labels even in non-convergence
        #===========================================================================
        #For now we will keep the obtained cluster centers even in case of non convergence to enable the automatic adjustment of preference
        labels = np.array([-1] * n_samples)
        #cluster_centers_indices = []
        modularity = -1

    if return_n_iter:
        if return_modularity:
            return cluster_centers_indices, labels, it + 1, modularity
        else:
            return cluster_centers_indices, labels, it + 1
    else:
        if return_modularity:
            return cluster_centers_indices, labels, modularity
        else:
            return cluster_centers_indices, labels


###############################################################################

class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor (between 0.5 and 1) is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, default=200
        Maximum number of iterations.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : bool, default=True
        Make a copy of input data.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        Which affinity to use. At the moment 'precomputed' and
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.

    verbose : bool, default=False
        Whether to be verbose.

    random_state : int, RandomState instance or None, default=0
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
    array and all training samples will be labelled as ``-1``. In addition,
    ``predict`` will then label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])
    """
    @_deprecate_positional_args
    def __init__(self, *, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False, random_state='warn',
                 adjacency=None, randomized_network=False, node_similarity_mode='shortest_path',
                 node_similarity_threshold=1, save_mask=False,
                 saving_path=None, enable_label_smoothing=True, smoothing_depth=1, dataset=None):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.adjacency = adjacency
        self.randomized_network = randomized_network
        self.node_similarity_mode = node_similarity_mode
        self.node_similarity_threshold = node_similarity_threshold
        self.save_mask = save_mask
        self.saving_path = saving_path
        self.enable_label_smoothing = enable_label_smoothing
        self.smoothing_depth = smoothing_depth
        self.dataset = dataset
    # TODO: Remove in 0.26
    # mypy error: Decorated property not supported
    @deprecated("Attribute _pairwise was deprecated in "  # type: ignore
                "version 0.24 and will be removed in 0.26.")
    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def _more_tags(self):
        return {'pairwise': self.affinity == 'precomputed'}

    def fit(self, X, y=None):
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        if self.affinity == "precomputed":
            accept_sparse = False
        else:
            accept_sparse = 'csr'
        X = self._validate_data(X, accept_sparse=accept_sparse)
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.cluster_centers_indices_, self.labels_, self.n_iter_, self.modularity_ = \
            affinity_propagation(
                self.affinity_matrix_, preference=self.preference,
                max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True, return_modularity=True,
                random_state=self.random_state,
                adjacency=self.adjacency, randomized_network=self.randomized_network,
                node_similarity_mode=self.node_similarity_mode,
                node_similarity_threshold=self.node_similarity_threshold, save_mask=self.save_mask,
                saving_path=self.saving_path, enable_label_smoothing=self.enable_label_smoothing,
                smoothing_depth=self.smoothing_depth, dataset=self.dataset)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        if self.cluster_centers_.shape[0] > 0:
            return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn("This model does not have any cluster centers "
                          "because affinity propagation did not converge. "
                          "Labeling every sample as '-1'.", ConvergenceWarning)
            return np.array([-1] * X.shape[0])

    def fit_predict(self, X, y=None):
        """Fit the clustering from features or affinity matrix, and return
        cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)