import pandas as pd
import numpy as np

def convertData(data):
  data.replace('Female', 0, inplace=True)
  data.replace('Male', 1, inplace=True)

def data_normalisation(data):
  
  for col in data.columns:
    data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

def removeOutliers(data):
  v = data.std()
  m = data.mean()
  threshold_values = []
  features_count = data.shape[1] - 1
  for i in range(features_count):
    threshold_values.append(2*m[i] + 5*v[i])
  for i in range(features_count):
    data.drop(data[data.iloc[:,i] >= threshold_values[i]].index, inplace = True)
  
def sample_split(data,split_size):
  train_data = data.sample(frac=split_size)
  test_data = data.drop(train_data.index)
  return train_data, test_data

class Bayesian_tree():
  def __init__(self,a=None):
    self.a = a

  def compute_prior(self,data,Y):
    features = sorted(pd.unique(pd.Series(Y)))
    prior = []
    for i in features:
      prior.append(data[Y == i].shape[0]/len(data))
    return prior
  
  def compute_likelihood(self,feature_index,feature_value,data,Y,label):
    data1 = data[Y==label]
    mean = data.iloc[:,feature_index].mean()
    std = data.iloc[:,feature_index].std()
    probability = (1 / (np.sqrt(2 * np.pi) * std))*np.exp(-((feature_value-mean)**2/(2*std**2)))
    return probability

  def compute_likelihood_using_laplacian(self, feature_index, feature_value, data,Y, label):
    data1 = data[Y==label]
    
    if feature_index != 1:
      mean = data.iloc[:,feature_index].mean()
      std = data.iloc[:,feature_index].std()
      probability = (1 / (np.sqrt(2 * np.pi) * std))*np.exp(-((feature_value-mean)**2/(2*std**2)))
      return probability
    else:
      if(feature_value):
        return 1-(data['gender'].sum()+1)/(len(data)+2)
      else:
        return (data['gender'].sum()+1)/(len(data)+2)

  def classifier_laplace(self,data,X,Y):
    features = list(X.columns)          # gives values like age, gender etc
    
    prior = self.compute_prior(data,Y)     # computes the value of p(is_patient), p(is_not_patient)
    
    Y_pred = []                           # to store the predicted values
    Z = X.to_numpy()
    for x in range(X.shape[0]):                # x = row count
      labels = sorted(pd.unique(pd.Series(Y)))  # finds all the values of is_patient i.e. 0, 1
      likelihood = [1]*len(labels)              # likelihood = [1 1]
      for j in range(len(labels)):              # j from 0 to 1
        for i in range(len(features)):          # i from 0 to 10 i.e. features count
          # Z = X.iloc[x:x+1,i].values()
          
          
          likelihood[j] *= self.compute_likelihood_using_laplacian(i, Z[x][i],data,Y,labels[j])
                                                # likelihood = [computed1 computed2]
      post_prob = [1]*len(labels)               
      for j in range(len(labels)):
          post_prob[j] = likelihood[j] * prior[j]
      Y_pred.append(np.argmax(post_prob))
    
    return np.array(Y_pred)
  

  def classifier(self,data,X,Y):

    features = list(X.columns)          # gives values like age, gender etc
    
    prior = self.compute_prior(data,Y)     # computes the value of p(is_patient), p(is_not_patient)
    
    Y_pred = []                           # to store the predicted values
    
    for x in range(X.shape[0]):                # x = row count
      labels = sorted(pd.unique(pd.Series(Y)))  # finds all the values of is_patient i.e. 0, 1
      likelihood = [1]*len(labels)              # likelihood = [1 1]
      for j in range(len(labels)):              # j from 0 to 1
        for i in range(len(features)):          # i from 0 to 10 i.e. features count
          likelihood[j] *= self.compute_likelihood(i, X.iloc[x:x+1,i],data,Y,labels[j])
                                                # likelihood = [computed1 computed2]
      post_prob = [1]*len(labels)               
      for j in range(len(labels)):
          post_prob[j] = likelihood[j] * prior[j]
      Y_pred.append(np.argmax(post_prob))
    
    return np.array(Y_pred)

out = open("q2_results.txt",'w')
df = pd.read_csv("Train_B_Bayesian.csv")
d1 = df.copy()
convertData(d1)
removeOutliers(d1)
data_normalisation(d1)
print("The final set of features after dropping the outliers in each column are ",file=out)
print(d1.columns,file=out)
K = 5
training_data,testing_data = sample_split(d1,.7)
Y_training = training_data.iloc[:,-1]
X_training = training_data.drop(training_data.columns[[-1]],axis=1)
Y_testing = testing_data.iloc[:,-1].values
X_testing = testing_data.drop(testing_data.columns[[-1]],axis=1)

a = Bayesian_tree()

diff = X_training.shape[0]//5
l = 0
r = 0

max_score = 0
best_training_data = 0

for j in range(K):
  
  l = r
  r = r + diff
  
  K_fold_testing_data = training_data.iloc[l:r]       # 5 fold cross validation test 
  
  K_fold_training_data = training_data.drop(K_fold_testing_data.index)
  
  Y_K_fold_training_data = K_fold_training_data.iloc[:,-1]
  
  X_K_fold_training_data = K_fold_training_data.drop(training_data.columns[[-1]],axis=1)
  
  Y_K_fold_testing_data = K_fold_testing_data.iloc[:,-1].values
  
  X_K_fold_testing_data = K_fold_testing_data.drop(K_fold_testing_data.columns[[-1]],axis=1)
  
  Y_K_fold_predicted_values = a.classifier(K_fold_training_data,X_K_fold_testing_data,Y_K_fold_training_data)
  
  positiveResults = np.sum(Y_K_fold_testing_data==Y_K_fold_predicted_values)
  
  TotalResults = Y_K_fold_testing_data.shape[0]
  
  curr_score = positiveResults/TotalResults
  
  if curr_score > max_score:
    Y_training = Y_K_fold_training_data
    X_training = X_K_fold_training_data
    best_training_data = K_fold_training_data
    max_score = curr_score


Y_predicted_values = a.classifier(best_training_data,X_testing,Y_training)
positiveResults = np.sum(Y_testing==Y_predicted_values)
TotalResults = Y_testing.shape[0]
print()
print("Accuracy using 5-fold cross validation : ",positiveResults/TotalResults,file=out)

# Laplacian
training_data,testing_data = sample_split(d1,.7)
Y_training = training_data.iloc[:,-1]
X_training = training_data.drop(training_data.columns[[-1]],axis=1)
Y_testing = testing_data.iloc[:,-1].values
X_testing = testing_data.drop(testing_data.columns[[-1]],axis=1)

Y_predicted_values = a.classifier_laplace(training_data,X_testing,Y_training)
positiveResults = np.sum(Y_testing==Y_predicted_values)
TotalResults = Y_testing.shape[0]
print()
print("Accuracy using Laplacian Classifier : ",positiveResults/TotalResults,file=out)
out.close()