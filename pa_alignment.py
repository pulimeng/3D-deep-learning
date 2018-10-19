def vrrotvec(a,b):
    """ Function to rotate one vector to another, inspired by
    vrrotvec.m in MATLAB """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a,b))
    angle = np.arccos(np.minimum(np.dot(a,b),[1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1,3))
        c[mind] = 0
        ax = normalize(np.cross(a,c))
    r = np.concatenate((ax,angle))
    return r

def vrrotvec2mat(r):
    """ Convert the axis-angle representation to the matrix representation of the 
    rotation """
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c
    
    n = normalize(r[0:3])
    
    x = n[0]
    y = n[1]
    z = n[2]
    
    m = np.array(
     [[t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
     [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
     [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]]
    )
    return m
        
        
inertia = np.cov(pocket_coords.T)
e_values, e_vectors = np.linalg.eig(inertia)
sorted_index = np.argsort(e_values)[::-1]
sorted_vectors = e_vectors[:,sorted_index]
# Align the first principal axes to the X-axes
rx = vrrotvec(np.array([1,0,0]),sorted_vectors[:,0])
mx = vrrotvec2mat(rx)
pa1 = np.matmul(mx.T,sorted_vectors)
# Align the second principal axes to the Y-axes
ry = vrrotvec(np.array([0,1,0]),pa1[:,1])
my = vrrotvec2mat(ry)
transformation_matrix = np.matmul(my.T,mx.T)
# transform the protein coordinates to the center of the pocket and align with the principal
# axes with the pocket
transformed_coords = (np.matmul(transformation_matrix,protein_coords.T)).T
