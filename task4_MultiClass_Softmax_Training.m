function W = trainMultiClassSoftmax(X,y,C)

    % << COMPLETE THE FUNCTION BODY! TYPICAL SOLUTION STEPS ARE GIVEN IN COMMENTS. >>

    % Initialize variables
    [P, N] = size(X);
    w0 = zeros(1, C * (N + 1));
    
    alpha = 1.0;
    max_iter = 1000;
    % Perform gradient descent on the cost_softmax function
    [~, w_best, ~, ~] = gradientDescentAD(@cost_softmax, w0, alpha, max_iter);  
    % Return the best weight vector but in matrix form
    W = reshape(w_best, [C, N + 1]);
    
    % Nested cost function
    function c = cost_softmax(w)

        % For computations, transform w into matrix form
        % Transform w into matrix form (C x N+1)
        W_mat = reshape(w, [C, N + 1]);

        % 1. Augment data with bias (x_hat)
        X_aug = [ones(P, 1), X]; % P x (N+1)

        % 2. Calculate scores for all classes: Score = X_hat * W^T
        
        all_scores = X_aug * W_mat';

        % 3. Evaluate the Multi-Class Softmax cost
        
        max_s = max(all_scores, [], 2);
        log_sum_exp = max_s + log(sum(exp(all_scores - max_s), 2));

        % Identify the score of the CORRECT class for each sample
        % y contains labels 0, 1, ..., C-1.
        correct_class_indices = sub2ind(size(all_scores), (1:P)', y + 1);
        correct_scores = all_scores(correct_class_indices);
        
        
        % Evaluate the Multi-Class Perceptron cost
        c = (1/P) * sum(log_sum_exp - correct_scores);% Complete the formula

    end

end
