function c = classifyMultiClass( W, X )
% CLASSIFYMULTICLASS Predicts class labels using the Multi-Class Fusion Rule
%   c = classifyMultiClass( W, X )
%
%   Implementation of the Fusion Rule from Lecture 5.
%   Ideally, we want to pick the class that yields the largest value 
%   for the decision rule: y' = argmax( x_hat * w' )
%
%   Inputs:
%       W: K x (N+1) weight matrix. Each ROW is the weight vector for one class.
%          (K = number of classes, N = number of features)
%       X: P x N data matrix. (P = number of samples)
%
%   Output:
%       c: P x 1 vector of predicted class labels (0 to K-1).

    % 1. Augment the Input Data
    % Add a column of ones (bias term) to the beginning of each sample.
    % X becomes P x (N+1)
    [P, ~] = size(X);
    X_aug = [ones(P, 1), X]; 
    
    % 2. Calculate Linear Scores (The Fusion Rule)
    % Compute the dot product for every sample against every class weight vector.
    % Dimensions: (P x N+1) * (N+1 x K) = P x K matrix
    scores = X_aug * W';
    
    % 3. Determine the Predicted Class (Argmax)
    % Find the index of the maximum score along the rows (dimension 2).
    [~, c_idx] = max(scores, [], 2);
    
    % 4. Adjust to 0-based indexing
    % The grader expects labels 0, 1, 2... but MATLAB indices are 1, 2, 3...
    c = c_idx - 1;

end
