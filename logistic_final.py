import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics

np.random.seed(12)
num_observations = 1500
test_obs = 500
x1 = np.random.multivariate_normal([1,0],[[1, 0.75],[0.75,1]], num_observations)
x2 = np.random.multivariate_normal([0,1.5],[[1, 0.75],[0.75,1]], num_observations)

xt1 = np.random.multivariate_normal([1,0],[[1, 0.75],[0.75,1]], test_obs)
xt2 = np.random.multivariate_normal([0,1.5],[[1, 0.75],[0.75,1]], test_obs)

simulated_separableish_features = np.vstack((x1,x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

test_simulated_separableish_features = np.vstack((xt1,xt2)).astype(np.float32)
test_simulated_labels = np.hstack((np.zeros(test_obs), np.ones(test_obs)))

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:,0], simulated_separableish_features[:,1], c = simulated_labels, alpha = 0.4)
plt.show()


plt.figure(figsize=(12,8))
plt.scatter(test_simulated_separableish_features[:,0], test_simulated_separableish_features[:,1], c = test_simulated_labels, alpha = 0.4)
plt.show()

def sigmoid(scores):
    return 1/(1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features,weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll

def cost_function(features, labels, weights):

    observations = len(labels)

    predictions = predict(features, weights)

    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost

def predict(features, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''
  z = np.dot(features, weights)
  return sigmoid(z)

def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 100 == 0:
            print ("Iteration",cost_function(features, target, weights))

    return weights

weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 3000, learning_rate = 0.0001, add_intercept=True)
data_with_intercept = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))
print ('Training Accuracy: {0}'.format((preds == simulated_labels).sum().astype(float) / (0.01 * len(preds))))

test_data_with_intercept = np.hstack((np.ones((test_simulated_separableish_features.shape[0], 1)),
                               test_simulated_separableish_features))

test_final_scores = np.dot(test_data_with_intercept, weights)

test_preds = np.round(sigmoid(test_final_scores))

print ('Testing Accuracy: {0}'.format((test_preds == test_simulated_labels).sum().astype(float) / (0.01 * len(test_preds))))

fpr, tpr, threshold = metrics.roc_curve(test_simulated_labels, test_preds)
roc_auc = metrics.auc(fpr, tpr)
# ROC Curve and AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.22f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()