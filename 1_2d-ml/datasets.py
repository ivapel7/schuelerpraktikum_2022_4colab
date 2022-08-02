import numpy as np
import sklearn.datasets

def generate_dataset(name='xor', D=2, N=2000, random_seed=1234):
    # (each internally called method) returns a triple:
    # X (the data points), Y (the labels), num_classes (an int communicating the number of classes generated from this dataset)
    
    
    np.random.seed(random_seed) # "Startwert" für zufalls generator -> reproduzierbare daten (denn nichts ist wirklich zufällig in der simulation/berechnung)
    # Generated datasets should be N x D - shaped, along with
    if name == 'xor':
        return data_xor(D,N)
    if name == 'four':
        return data_four(D,N)
    if name == 'three':
        return data_three(D,N)
    if name == 'circles':
        return data_circles(D,N)
    if name == 'moons':
        return data_moons(D,N)
    else:
       raise ValueError('Unbekannter Datensatzname "{}"'.format(name))

def data_moons(D,N):
    X, Y = sklearn.datasets.make_moons(n_samples=N, noise=0.05)
    Y = np.array(Y) == 1
    Y = (np.vstack((Y, np.invert(Y)))*1.0).T # and [NxC] labels
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, 2
    
def data_circles(D,N):
    X, Y = sklearn.datasets.make_circles(n_samples=N, noise=0.05)
    Y = np.array(Y) == 1
    Y = (np.vstack((Y, np.invert(Y)))*1.0).T # and [NxC] labels
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, 2

def data_three(D,N):
    X, Y = sklearn.datasets.make_blobs(n_samples=N, centers=3, n_features=2, random_state=1)
    X = np.array(X)
    Y0 = np.array(Y) == 0
    Y1 = np.array(Y) == 1
    Y2 = np.array(Y) == 2
    Y = (np.vstack((Y0, Y1, Y2))*1.0).T # and [NxC] labels
    return X, Y, 3

def data_xor(D,N):
    #this is the XOR problem.
    X = np.random.rand(N,D) #we want [NxD] data
    X = (X > 0.5)*1.0
    Y = X[:,0] == X[:,1]
    Y = (np.vstack((Y, np.invert(Y)))*1.0).T # and [NxC] labels

    X += np.random.randn(N,D)*0.1 # add some noise to the data.
    return X, Y, 2

def data_four(D,N):
    #this is the XOR problem.
    X = np.random.rand(N,D) #we want [NxD] data
    X = (X > 0.5)*1.0
    Y0 = (X[:,0] == 0) * (X[:,1] == 0)
    Y1 = (X[:,0] == 1) * (X[:,1] == 0)
    Y2 = (X[:,0] == 0) * (X[:,1] == 1)
    Y3 = (X[:,0] == 1) * (X[:,1] == 1)

    Y = (np.vstack((Y0, Y1, Y2, Y3))*1.0).T # and [NxC] labels
    X += np.random.randn(N,D)*0.1 # add some noise to the data.
    return X, Y, 4
    

def data_eval(X, n_per_dim=100, border=0.1):
    # generate test grid data matching the training data, with a bit of extra space around
    assert X.shape[1] == 2, "X muss in R^2 sein, um grafisch evaluiert zu werden."
    
    # analyze min/max values
    xmin = np.amin(X, axis=0)
    xmax = np.amax(X, axis=0)
    val_range = np.abs(xmax-xmin)

    # create exhaustive coordinate grid
    x = np.linspace(xmin[0] - val_range[0]*border, xmax[0] + val_range[0]*border, n_per_dim)
    y = np.linspace(xmin[1] - val_range[1]*border, xmax[1] + val_range[1]*border, n_per_dim)
    xx, yy = np.meshgrid(x, y)
    
    #return as 2d-Cordinates and grid coordinates
    Xval = np.concatenate([xx[...,None], yy[...,None]], axis=2)
    Xval = np.reshape(Xval, (n_per_dim**2, 2))
    return Xval, (xx, yy)