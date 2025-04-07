import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def load_pf(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
        X = []
        F = []
        for line in data:
            x_vals, f_vals = line.split(':')
            X.append(np.fromstring(x_vals, sep=','))
            F.append(np.fromstring(f_vals, sep=','))
            
        return np.array(F), np.array(X)
    
def save_pf(fname, F, X):
    with open(fname, 'w') as f:
        for i in range(F.shape[0]):
            row_X = ','.join(map(str, X[i]))
            row_F = ','.join(map(str, F[i]))
            f.write(f"{row_X}:{row_F}\n")
    
# for this function we assume the PF is at 0.5 x=for all but the first variable (or the variable specified by x0)
def get_test_fn_pf(test_prob, n_pareto_points=500, initial_multiplier=5, x0=0.5, fill_missing_points=True):        
    x1 = np.linspace(test_prob.xl[0], test_prob.xu[0], n_pareto_points*initial_multiplier)
    x2 = x0 * np.ones((n_pareto_points*initial_multiplier, test_prob.n_var - 1))
    X = np.column_stack((x1, x2))
    out = {}
    test_prob._evaluate(X,out)
    F = out["F"]
    pf_ids = select_by_ref_vector(F, n_pareto_points)
    if len(pf_ids) < n_pareto_points and fill_missing_points:
        pf_ids = select_by_max_min_distance(F, n_points=n_pareto_points, preselected=pf_ids)
    F = F[pf_ids]
    X = X[pf_ids]
    sorted_ids = np.argsort(F[:,0])
    F = F[sorted_ids]
    X = X[sorted_ids]
        
    return F, X


def perpendicular_distance(point, r1, r2):
    line_vec = r2 - r1
    point_vec = point - r1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unitvec)
    closest_point_on_line = r1 + projection_length * line_unitvec
    return np.linalg.norm(point - closest_point_on_line)


def select_by_ref_vector(F, n_points=100):
    # sort by f1
    sorted_ids = np.argsort(F[:,0])
    F = F[sorted_ids]
    
    # normalise the objective space
    F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    
    line_points = np.linspace(F[0], F[-1], n_points)
    # add extreme points to the list of closest points
    closest_ids = [0, len(F) - 1]
    
    # do sparse selection by reference vectors for double the required points
    # because points are sorted, we only need to start at the point closest to the previous reference vector
    # and we can stop looking once the distance increases
    current_id = 0
    for i in range(n_points):
        dists = [np.inf]
        for p_id in range(current_id, F.shape[0]):
            dist = perpendicular_distance(F[p_id], line_points[i], -line_points[n_points-i-1])
            if dist < dists[-1]:
                dists.append(dist)
                current_id = p_id
            else:
                break
        
        closest_ids.append(current_id)
        
    sparse_ids = np.array(list(set(closest_ids)))
    
    return sorted_ids[sparse_ids]


def select_by_max_min_distance(F, n_points, preselected=[]):
    if F.shape[0] <= n_points:
        return np.arange(F.shape[0])
    
    # normalise the objective space
    F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    
    # Start with extreme points or any preselected points
    sparse_ids = list(set(np.concatenate(([np.argmin(F[:, 0]), np.argmin(F[:, 1])], preselected))))
    selected_mask = np.zeros(F.shape[0], dtype=bool)  # Boolean mask to track selected points
    selected_mask[sparse_ids] = True

    # Get the initial set of non-selected points
    non_selected = np.where(~selected_mask)[0]
    num_non_selected = len(non_selected)
    
    # Initialize distances matrix
    distances = np.full((len(sparse_ids), num_non_selected), np.inf)
    
    for i, s_id in enumerate(sparse_ids):
        distances[i, :] = np.linalg.norm(F[s_id] - F[non_selected], axis=1)
        
    n_to_select = n_points - len(sparse_ids)
    # Continue until we have enough points
    
    while len(sparse_ids) < n_points:
        # Find the maximum of the minimum distances
        min_dists = np.min(distances, axis=0)
        next_index_in_non_selected = np.argmax(min_dists)
        next_index = non_selected[next_index_in_non_selected]

        # Add the next point to the selection
        sparse_ids.append(next_index)
        selected_mask[next_index] = True
        
        # Update distances: Set distance to already selected points as -inf
        new_distances = np.linalg.norm(F[next_index] - F[non_selected], axis=1)
        distances = np.vstack((distances, new_distances))
        distances[-1, np.isin(non_selected, sparse_ids)] = -np.inf
        
    return np.array(sparse_ids)


def get_ND_set_by_batches(F, batch_size=20000):
    n_batches = (F.shape[0] // batch_size) + 1
    ND_ids = np.array([], dtype=int)
    for i in range(n_batches):
        # Calculate the batch slice
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, F.shape[0])

        # Create a batch by combining current ND set and new batch
        batch_F = np.vstack([F[ND_ids], F[batch_start:batch_end]])
        
        # Find unique points in the batch
        unique_F, unique_ids = np.unique(batch_F, axis=0, return_index=True)
        
        # Perform non-dominated sorting
        batch_nd_ids = NonDominatedSorting().do(unique_F, only_non_dominated_front=True)
        
        # Map back to original indices
        batch_nd_ids = unique_ids[batch_nd_ids]
        
        # Separate old and new indices
        old_ids = batch_nd_ids[batch_nd_ids < len(ND_ids)]
        new_ids = batch_nd_ids[batch_nd_ids >= len(ND_ids)] - len(ND_ids) + batch_start

        # Update ND_ids
        ND_ids = np.concatenate([ND_ids[old_ids], new_ids]).astype(int)
        
        print(f"Batch {i + 1}/{n_batches} - Number of solutions in the ND set: {len(ND_ids)}")

    return ND_ids.astype(int)
