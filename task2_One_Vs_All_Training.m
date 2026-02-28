function W = trainOneVsAll( X, y, C )

    % << IMPLEMENT FUNCTION BODY. SOME USUAL MAJOR STEPS ARE GIVEN IN THE COMMENTS BELOW >>

    % Initialize variables
    [P, N] = size(X);
    W = zeros(C, N + 1);
    % Perform One-vs-All: Train each class against all the others one by one
    for i = 1:C
        
        % Form the two-class problem
        current_class = i - 1;
        
        y_binary = -ones(P, 1);
        y_binary(y == current_class) = 1;
        % Use trainPerceptron on the two-class problem
        w_binary = trainPerceptron(X, y_binary);
        % Store the best weight
        W(i,:) = w_binary; % pick the best weight here
        
    end

    % Normalize weights
    for i = 1:C
        row_norm = sqrt(sum(W(i, :).^2));
        if row_norm > 0
            W(i, :) = W(i, :) / row_norm;
        end
    end
end
