
import numpy as np

""" *************************** COSTS & GRADIENTS *********************************** """

"""MSE"""
def MSE(y, tx, w):
    return np.mean( np.square( np.subtract(y,np.dot( tx , w )) ) )/2

def MSE_grad(y, tx, w):
    error = np.subtract(y,np.dot(tx,w))
    return -np.mean(tx * error[:, None], axis = 0) / 2

"""MAE"""
def MAE(y, tx, w):
    error = np.subtract(y,np.dot(tx,w))
    return np.mean(np.abs(error), axis = 0)

def MAE_grad(y, tx, w):
    error = np.subtract(y,np.dot(tx,w))
    return -np.mean(np.sign(error[:, None]) * tx, axis = 0)
    
"""Calculate the loss."""
def compute_loss(y, tx, w, method = MSE):
    return method(y,tx,w)

""" *************************** GRID SEARCH *********************************** """
def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1

def grid_search(y, tx, num_intervals = 50, method = MSE, last = False):
    
    # Generate w0, w1
    w = generate_w(num_intervals)
    d = w[0].shape[0]
    
    # Get K combinations of w0,w1 => [D x K]
    rows, columns = np.indices((d, d))
    w0, w1 = w[0][rows.ravel()], w[1][columns.ravel()]
    ws = np.array([w0,w1])
    
    # Compute loss => [K, ]
    losses = compute_loss(np.expand_dims(y, axis=1), tx, ws, method)
    
    # losses [K, ] => grid_losses [DxD]
    grid_losses = np.reshape(losses, (d,d))
    
    # Get optimal 
    loss_star, w0_star, w1_star = get_best_parameters(w[0], w[1], grid_losses)
    
    if last: return [loss_star, np.array([w0_star, w1_star])]
    else: return [grid_losses, w[0], w[1], loss_star, w0_star, w1_star]
    

def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


""" *************************** SOLVERS *********************************** """

""" *** Given method, assign method_grad *** """
def assign_grad(method):
    if method.__name__ == MSE.__name__: 
        return MSE_grad
    elif method.__name__ == MAE.__name__:
        return MAE_grad
    else: raise NotImplementedError
        
""" ******** GD ******** """

def compute_gradient(y, tx, w,
                     method_grad = MSE_grad):
    """Compute the gradient."""
    return method_grad(y, tx, w)


def gradient_descent(y, tx, initial_w,
                     max_iters, gamma, method = MSE, last = False):
    """Gradient descent algorithm."""
    
    #Define grad function
    method_grad = assign_grad(method)
    
    #initiate w
    w = initial_w

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, w, method)]
    #iterations
    for n_iter in range(max_iters):        
        # Compute gradient
        grad = compute_gradient(y, tx, w, method_grad)

        # Update model parameters
        w = w - gamma * grad
        
        # Store w and loss
        if(last):
            ws.append(w)
            losses.append(compute_loss(y, tx, w, method))

    if last : return w, compute_loss(y, tx, w, method)
    else : return ws, losses

        
        
"""" ******** SGD ******** """
#compute the stochastic gradient from least square
#mini batch size (1)
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    grads = []
    #compute each individual gradient, then store them
    for i in range(len(y)):       
        grads.append( MSE_grad(np.array([y[i]]), np.array([tx[i]]), w) )
    #compute the mean of all the singular gradients    
    return np.mean(grads,axis=0)

#least square stochastic gradient descent
def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w

    #w = initial_w
    for n_iter in range(max_iters):
        #compute gradient
        grad = compute_stoch_gradient(y, tx, w)
        #apply gradient to w
        w =  w - gamma*grad 
    
    return  w, MSE(y, tx, w)
    
"""" ******** LEAST SQUARES ******** """

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # Get gram matrix = X.T * X
    gram = tx.T.dot(tx)
    sol = tx.T.dot(y)
    w = np.linalg.solve(gram, sol)
    #loss (mse)
    mse = compute_loss(y, tx, w)
    # ***************************************************
    return w, mse   

"""" ******** RIDGE-REGRESSION ******** """

def ridge_regression(y, tx, lambda_ ):
    """calculate the ridge regression solution for a given lambda."""
    xtx = np.dot(tx.T,tx)
    w = np.linalg.solve( np.add(xtx,lambda_*np.ones(xtx.shape))  , np.dot(tx.T,y) )
    return w , compute_loss(y,tx,w)
    
"""" ******** LOGISTIC REGRESSION ******** """

def sigmoid(t):
    exp = np.exp( t )
    return exp/(1+exp)

def compute_gradient_sigmoid(y, tx, w ):
    return np.dot( tx.T , sigmoid( np.dot(tx,w)  ) - y )

def compute_loss_sigmoid(y, tx, w ):
    txw = sigmoid(np.dot(tx,w))
    return -(y * np.log(txw) + (1-y) * np.log(1-txw)).sum()
    #return np.sum( np.log( 1 + np.exp(txw) ) - np.dot(y,txw) ) 

def compute_loss_sigmoid_MSE(y,tx,w):
    return np.mean( np.square(y-sigmoid(tx @ w)) )/2


#logistic reg gradient descent 
def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """logistic regression gradient descent algorithm."""
    #initiate w
    w = initial_w
    
    for i in range(max_iters):
        #compute gradient
        grad = compute_gradient_sigmoid(y, tx, w )
        #apply gradient to w
        w = w - gamma * grad
        
    return w , compute_loss_sigmoid(y, tx, w )

#logistic reg stochastic gradient descent 
def logistic_regression_stoch(y, tx, initial_w,max_iters, gamma,batch_size = 1,seed=42):
    """logistic regression gradient descent algorithm."""
    #initiate w
    w = initial_w

    #creation of the batches
    idx = np.arange( len(y) )

    np.random.seed(seed)
    np.random.shuffle(idx)

    batches_tx = []
    batches_y  = []

    n=len(y)//batch_size
    for i in range(n):
        batches_tx.append( tx[ i*batch_size : (i+1)*batch_size , : ]  )
        batches_y.append( y[ i*batch_size : (i+1)*batch_size ]  )

    ###
    
    for i in range(max_iters):
        #compute gradient
        grads = []
        for j in range(n):
            grads.append( compute_gradient_sigmoid( batches_y[j], batches_tx[j], w ) )
    
        #apply gradient to w
        w = w - gamma * np.mean(grads,axis=0)
        
    return w , compute_loss_sigmoid(y, tx, w )

#regularized logistic regression
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):

    #how to set the threshold ? testing ? paramater ?
    threshold = 1e-8
    #initiate w
    w = initial_w
    loss1=0
    loss2=0

    for i in range(max_iters):
        #compute gradient
        grad = compute_gradient_sigmoid(y, tx, w )
        #apply gradient to w
        w = w - gamma * grad
        #       classical loss                 penality
        loss2 = compute_loss_sigmoid(y,tx,w) + lambda_*np.linalg.norm(w)**2

        #trigger security break if threshold is passed
        if( (i>0) and ( abs(loss1 - loss2) < threshold ) ):
            print( 'RLR stop at step ' + str(i) )
            break
        loss1=loss2
        
    return w , compute_loss_sigmoid(y, tx, w )
    
"""  *************************** LOAD DATA *********************************** """

def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metrics system."""
    path_dataset = "./data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender

""" *************************** STANDARDIZE DATA *********************************** """
# standardize the data ( (data-mean)/std )
def standardize(data):
    mean = np.mean(data, axis=0)
    centered = (data - mean)
    std = np.std(centered, axis=0)
    
    data_s = centered/std
    
    return data_s

""" *************************** ERROR REPLACEMENT *********************************** """
# Get overview of data
def show(data, rows = 5, width = 4):
    for col in range (0, data.shape[1], width):
        print('col =',str(col),':',str(col+width),'\n', data[0:rows, col:col+width], '\n')
        
# Replace 
def replaceByMedian(data, replace,display=False):
    median = np.nanmedian(np.where(data == replace, np.nan, data), axis = 0)
    
    for col in range (data.shape[1]):
        extract = data[:,col]
        data[:,col] = np.where(extract == replace, median[col], extract)
    
    if(display):show(data, rows = 2)

""" *************************** SPLIT DATA *********************************** """

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    nb_rows = len(y)
    
    indices = np.random.permutation(nb_rows)
    splitAt = int(np.floor(ratio * nb_rows))
    
    index_tr = indices[:splitAt]
    index_te = indices[splitAt:]
    
    x_tr = x[index_tr]
    y_tr = y[index_tr]
    
    x_te = x[index_te]
    y_te = y[index_te]
    # ***************************************************
    return x_tr, y_tr, x_te, y_te


""" *************************** EXPAND DATA *********************************** """
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


""" *************************** CORRELATION *********************************** """

def correlation_pearson(X, y = None):
    """ return Pearson's correlation coeff for a given feature matrix X and prediction variable y"""
    # X : feature matrix n * m , n datapoints, m features
    # y : prediction variable, size n , n values, n datapoints
    Pearson = np.corrcoef(X, y = y, rowvar = False)
    return Pearson

def correlation_spearman(X, y = None):
    """ return Spearman's correlation coeff for a given feature matrix X and prediction variable y"""
    # X : feature matrix n * m , n datapoints, m features
    # y : prediction variable, size n , n values, n datapoints
    Spearman = np.corrcoef(np.argsort(X.T,axis = 0), y = y)
    return Spearman

""" *************************** CROSS VALIDATION *********************************** """
# n -> number of groups
# loss_fun -> the function to compute loss (for test set)
# method -> the leaning method used on training set (least suaqres, graient descent etc)
# *args -> the argument used in method ( after y,tx )
def cross_validation(y, tx, n, loss_fun , method ,  seed = 42 ,kwargs={} ):
    #set seed, and mix indices
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    
    #create n groups
    split = int( num_row / n )
    groups = [ indices[split*i:split*(i+1)] for i in range(n) ]
    
    losses_train = []
    losses_test  = []
    ws = []
    
    #iterate each time excluding 1 group from training set and using it as test set
    for i in range(n):
        id_test = groups[i]    
        ## set data sets
        #Test
        xTest = tx[id_test]
        yTest = y[id_test]
        #Train
        xTrain = tx[~id_test]
        yTrain = y[~id_test]
        
        # training of the model
        w,l = method( yTrain,xTrain, **kwargs )
        
        #loss for the train set
        loss_train = loss_fun( yTrain, xTrain , w )
        #loss for the test set
        loss_test = loss_fun( yTest, xTest , w )
        
        #store results
        losses_train.append(loss_train)
        losses_test.append(loss_test)
        ws.append(w)
    
    return losses_train, losses_test, ws

""" *************************** PSEUDO 2D KNN USING GRID *********************************** """
#this function create a linear grid on the values of 2 features and associated each box with the probability
#of having y = 1. this probability can be smooth by extending the zone to neigbor boxes (smooth parameter)
#n is the number of valeus for intervals(so we got n-1 intervals for each features)
#return the grid of probabily and the vector of features interval
def knn_grid_smooth( var1, var2 , y , n = 10 , smooth=1 ):
    # create vector to cut both varaibles into linear pieces
    vec1 = np.linspace(start=np.min(var1), stop=np.max(var1), num=n, endpoint=True,)
    vec2 = np.linspace(start=np.min(var2), stop=np.max(var2), num=n, endpoint=True,) 
    #grid to store result according to the position
    grid = np.zeros( (n-1,n-1) )
    #regularization by the number of point to avoid outliers
    max_pts = 0
    #masks are used to get the set of point in the wanted subzone
    for i in range(len(vec1)-1):
        mask1 =  (var1 > vec1[ max( [i-smooth,0] ) ]) & (var1 < vec1[ min( [i+1+smooth,len(vec1)-1] ) ])
        for j in range(len(vec2)-1):
            mask2 =  (var2 > vec2[ max( [j-smooth,0] ) ]) & (var2 < vec2[ min( [j+1+smooth,len(vec2)-1] ) ])
            mask = (mask1 & mask2)
            #get the predictions for the subzone
            tmp_y = y[mask]
            #length/width -> used to regularize (otherwise edges are not scaled)
            diffI = min( [i+1+smooth,len(vec1)-1] ) - max( [i-smooth,0] ) 
            diffJ = min( [j+1+smooth,len(vec2)-1] ) - max( [j-smooth,0] )
            #compute the regularized score
            if( len(tmp_y)>0 ):
                mean = ( (tmp_y == 1).sum() ) / (diffI*diffJ) 
            else: mean = 0.0 #default value is 0
            grid[i,j] = mean
            #find the maximum of point for 1 subzone
            if(len(tmp_y) > max_pts ): max_pts=len(tmp_y)
    #normalization
    grid = grid/max_pts
    grid /= np.max(grid)
    return grid,vec1,vec2

# TEST -> predict the outcome from the grid of probability
# given a grid of probability and the intervals vecotr associated
# return the probability associate to the position of point on the grid
# (basically the reverse process of previous function)
def knn_grid_predict( var1, var2 , grid, vec1, vec2 ):
    #create the vector to return
    #default value is zero
    result = np.zeros( len(var1) )
    #find the points in the different defined subzone
    #and attribute them the associated value in grid
    for i in range(len(vec1)-1):
        mask1 =  (var1 > vec1[i]) & (var1 < vec1[i+1])
        for j in range(len(vec2)-1):
            mask2 =  (var2 > vec2[j]) & (var2 < vec2[j+1])
            mask = 1*(mask1 & mask2)
            result += mask * grid[i,j]
    return result

#Get tuples of indices of features that have high correlation
#Input a matrix of absolute values, does not take into account negative values
def getIndexHighCorr(X, thr = 0.8):
    # X : Correlation matrix
    indexList = []
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            if i < j :
                if X[i, j] > thr :
                    indexList.append([i, j])
    return np.array(indexList)
    