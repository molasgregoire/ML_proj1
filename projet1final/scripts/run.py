# -*- coding: utf-8 -*-
#%%
# Libraries and imports
import numpy as np

#imports
from implementations import *
from proj1_helpers import *
#%%
#load training data set
DATA_TRAIN_PATH = './../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample = False)

#load test data set
DATA_TEST_PATH = './../data/test.csv'
_, X_test, ids_test = load_csv_data(DATA_TEST_PATH)


### 1_ Preprocessing
# copy dataset
my_tx = tX.copy()

## extract PRI_jet_num from my_tx (because it is a classifier feature)
PRI_jet_num = my_tx[:,22]
my_tx = np.c_[my_tx[:,:22], my_tx[:,23:]]

###TEST
PRI_jet_num_test = X_test[:,22]
X_test = np.c_[X_test[:,:22], X_test[:,23:]]

## missing data management
# A look at nan values
mask = (my_tx == -999)
nb_rows = my_tx.shape[0]
emptyRate = 100*np.sum(mask / nb_rows, axis =0)

# get rid of columns with more than 70% of empty 
mask = np.sum(mask/nb_rows, axis =0) < 0.7
lt70 = my_tx[:, mask]
###TEST -> apply same mask
X_test_70 = X_test[:, mask]

# create a new feature: "# of missing values"
nb_missing = (lt70 == -999).sum(axis = 1)
###TEST
nb_missing_test = (X_test_70 == -999).sum(axis = 1)

#replace missing values by the median of the feature
replaceByMedian(lt70, -999)
### TEST
replaceByMedian(X_test_70, -999)

#standardize the data set (to have a similar scale on each feature)
lt70_stdz = standardize(lt70)
### TEST
X_test_70_stdz = standardize(X_test_70)

## create a new feature to approximate the pattern between the features 14 and 16
imax = 0.78
jmax = 2.72
f14Xf16 = np.array(1* (abs(lt70_stdz[:,14]-lt70_stdz[:,16]) < imax ))
f14Xf16 += np.array(1* (abs(lt70_stdz[:,14]-lt70_stdz[:,16]) > jmax ))
### TEST
f14Xf16_TEST = np.array(1* (abs(X_test_70_stdz[:,14]-X_test_70_stdz[:,16]) < imax ))
f14Xf16_TEST += np.array(1* (abs(X_test_70_stdz[:,14]-X_test_70_stdz[:,16]) > jmax ))

# using abolute values of 14 and 16 to centralize the distribution
# then compute the ditance from the diagonale
f14Xf16_abs =  np.abs( np.abs(lt70_stdz[:,14]) - np.abs(lt70_stdz[:,16]) )
f14Xf16_abs_TEST =  np.abs( np.abs(X_test_70_stdz[:,14]) - np.abs(X_test_70_stdz[:,16]) )

## use pseudo KNN on grid on pairs of variables found to have a visible pattern

# 14 and 16 interaction
grid,vec1,vec2=knn_grid_smooth( (lt70_stdz[:,14]) , (lt70_stdz[:,16]) , y  , n = 100,smooth=3)
knn_14_16 = knn_grid_predict( (lt70_stdz[:,14]) , (lt70_stdz[:,16]) , grid, vec1, vec2 )
###TEST
knn_14_16_test = knn_grid_predict( (X_test_70_stdz[:,14]) , (X_test_70_stdz[:,16]) , grid, vec1, vec2 )

# 3 and 4 interaction
grid,vec1,vec2=knn_grid_smooth( (lt70_stdz[:,3]) , (lt70_stdz[:,4]) , y  , n = 100,smooth=3)
knn_3_4 = knn_grid_predict( (lt70_stdz[:,3]) , (lt70_stdz[:,4]) , grid, vec1, vec2 )
###TEST
knn_3_4_test = knn_grid_predict( (X_test_70_stdz[:,3]) , (X_test_70_stdz[:,4]) , grid, vec1, vec2 )

# 4 and 6 interaction
grid,vec1,vec2=knn_grid_smooth( (lt70_stdz[:,4]) , (lt70_stdz[:,6]) , y  , n = 100,smooth=3)
knn_4_6 = knn_grid_predict( (lt70_stdz[:,4]) , (lt70_stdz[:,6]) , grid, vec1, vec2 )
###TEST
knn_4_6_test = knn_grid_predict( (X_test_70_stdz[:,4]) , (X_test_70_stdz[:,6]) , grid, vec1, vec2 )


### Creation of the final datasets
## basic features selction -> keep only the features found to be interesting
lt70_new_features = (lt70_stdz.copy())[:,[0,1,2,4,7,8,9]]
X_test_new_features = (X_test_70_stdz.copy())[:,[0,1,2,4,7,8,9]]

## reintroduce PRI_jet_num
# we melt PRI_jet_num along its value as it's a classifier
for i in np.unique(PRI_jet_num):
    tmp = np.array(1*( PRI_jet_num == i ))
    #test
    tmp = np.array(1*( PRI_jet_num_test == i ))
  
## Add # of missing values
lt70_new_features = np.c_[ lt70_new_features , nb_missing ]
###TEST
X_test_new_features = np.c_[ X_test_new_features , nb_missing_test ]

## Add probability heatmap for 14 16
lt70_new_features = np.c_[ lt70_new_features , knn_14_16 ]
#test
X_test_new_features = np.c_[ X_test_new_features , knn_14_16_test ]

## Add probability heatmap for 3 4
lt70_new_features = np.c_[ lt70_new_features , knn_3_4 ]
#test
X_test_new_features = np.c_[ X_test_new_features , knn_3_4_test ]

## Add probability heatmap for 4 6
lt70_new_features = np.c_[ lt70_new_features , knn_4_6 ]
#test
X_test_new_features = np.c_[ X_test_new_features , knn_4_6_test ]

## Add a column of ones
lt70_new_features = np.c_[ lt70_new_features , np.ones(len(lt70_new_features)) ]
#test
X_test_new_features = np.c_[ X_test_new_features , np.ones(len(X_test_new_features)) ]

#%%
### Model testing

#the stochastic logistic regression here is made to work as a clasifier for {0,1}
#then, create a new y to pass form -1/1 to 0/1
new_y = (y+1)/2

#cross validation test
# stochastic logistic regression, cross validation (10 groups)
losses_train, losses_test, ws = cross_validation(y=new_y, tx=lt70_new_features,
 n=10, loss_fun = compute_loss_sigmoid , 
method = logistic_regression_stoch,  seed = 42 ,
kwargs={'initial_w':[0]*lt70_new_features.shape[1],'max_iters':10**3, 'gamma':10**-0} )
#%%
## generate the prediction on the test set

pred_lr = sigmoid( X_test_new_features @ np.mean(ws,axis=0) )
pred_lr_absolute = 1*(pred_lr > 0.5)
pred_lr_absolute = (2*pred_lr_absolute)-1
create_csv_submission( ids_test , pred_lr_absolute , 'log_reg_SGD.csv')


#%%
pred_lr = sigmoid( lt70_new_features @ np.mean(ws,axis=0) )
pred_lr_absolute = 1*(pred_lr > 0.5)
pred_lr_absolute = (2*pred_lr_absolute)-1
diff = y - pred_lr_absolute 
print(np.unique(diff, return_counts=True)[1][1]/len(new_y))


#%%




