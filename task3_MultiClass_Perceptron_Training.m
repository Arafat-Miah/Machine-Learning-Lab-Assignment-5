function W = trainMultiClassPerceptron(X,y,C)

    % << COMPLETE THE FUNCTION BODY! TYPICAL SOLUTION STEPS ARE GIVEN IN COMMENTS. >>

    % Initialize variables
    [P, N] = size(X);
    w0 = zeros(1, C * (N+1));
    
    alpha = 1.0;
    max_iter = 1000;
    % Perform gradient descent on the cost_perceptron function
     [~, w_best, ~, ~] = gradientDescentAD(@cost_perceptron, w0, alpha, max_iter);   
    % Return the best weight vector but in matrix form
 W = reshape(w_best, [C, N + 1]);
    
    % Nested cost function
    function c = cost_perceptron(w)

        % For computations, transform w into matrix form
        W_mat = reshape(w, [C, N + 1]);
        % 1. Augment data with bias
        X_aug = [ones(P, 1), X]; % P x (N+1)

        % 2. Calculate scores for all classes: Score = X_hat * W^T
        
        all_scores = X_aug * W_mat';

        % 3. Identify the score of the CORRECT class for each sample
       
        correct_class_indices = sub2ind(size(all_scores), (1:P)', y + 1);
        correct_scores = all_scores(correct_class_indices);

        % 4. Evaluate the Multi-Class Perceptron cost
        
        max_scores = max(all_scores, [], 2);
        
        % The Perceptron cost is the difference between the highest score 
        
        c = (1/P) * sum(max_scores - correct_scores);

    end

end
