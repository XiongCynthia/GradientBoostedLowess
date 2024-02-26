import torch
from RegressionTree import RegressionTree
from Lowess import Lowess

# *Now* we make the gradient boosted lowess regression
class GradientBoostedLowess:
    '''
    A class for fitting and predicting data on a LOWESS  model with gradient boosting.
    '''
    def __init__(self, learning_rate=0.1, n_estimators=100, 
                 kernel=None, tau=0.05, 
                 min_samples_split=20, max_depth=5):
        '''
            Args:
                learning_rate: The amount of contribution each regression tree has.
                n_estimators: The number of boosting stages to perform.

                kernel: Kernel smoothing function for LOWESS. If None, Gaussian kernel
                smoothing will be used by default.
                tau: Bandwidth for LOWESS.

                min_samples_split: The minimum number of observations in a sample that 
                can be split in the decision trees.
                max_depth: Maximum depth of the decision trees.
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kernel = kernel
        self.tau = tau
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        return
        

    def fit(self, X:torch.Tensor, y:torch.Tensor):
        '''
        Fits data on a gradient boosted LOWESS model.

        Args:
            X: Training data
            y: Target values
        '''
        # Fit and predict with the LOWESS model
        if self.kernel is None:
            lowess = Lowess(tau=self.tau)
        else:
            lowess = Lowess(kernel=self.kernel, tau=self.tau)
        lowess.fit(X.numpy(), y.numpy())
        lowess_pred = torch.Tensor(lowess.predict(X.numpy()))
        resids = y - lowess_pred
        self.lowess_ = lowess

        # Use regression trees to predict residuals
        self.trees_ = []
        for estimator in range(self.n_estimators):
            tree = RegressionTree(min_samples_split=self.min_samples_split, 
                                  max_depth=self.max_depth)
            tree.fit(X, resids)
            resids = y - (lowess_pred + tree.predict(X)*self.learning_rate)
            self.trees_.append(tree)
        return
    
    
    def predict(self, X:torch.Tensor):
        '''
        Predicts data using the fitted gradient bossted LOWESS model.

        Args:
            X: Sample data
        '''
        if not self.is_fitted():
            raise Exception("This GradientBoostedLowess instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        y_preds = []
        for obs in X:
            if obs.shape[0] == 1: # If the fitted dataset has only 1 feature
                pred = self.lowess_.predict(obs.numpy())[0] # Lowess class is not compatible with pyTorch tensors
            else:
                pred = self.lowess_.predict([obs.numpy()])[0]
            
            for tree in self.trees_:
                if obs.shape[0] == 1:
                    pred += tree.predict(obs)*self.learning_rate
                else:
                    pred += tree.predict(obs.reshape(1,-1))

            y_preds.append(pred)
        return torch.Tensor(y_preds)
    
    def is_fitted(self):
        if hasattr(self, 'lowess_') and hasattr(self, 'trees_'):
            return True
        else:
            return False
