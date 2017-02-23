function trans_matrix = imageMatching(match, pos1, pos2)
%     purpose: find pairwise alignment, i.e the relation between two pictures
%     how to:  solve linear systems A\B or using RANSAC
%     input:   matching_point pairs index
%     output:  matrix M representing relation between two pictures    
    % sovle linear system
    
    n = size(match, 1);
    A = zeros(n*2+1, 3);
    b = zeros(n*2+1, 1);

    % putting x to A and b
    for i = 1:n
        A(i, 1) = 1;
        A(i, 3) = pos1(match(i, 1), 2);
        b(i, 1) = pos2(match(i, 2), 2);
    end
    
    % putting y to A and b
    for i = 1:n
        j = i + n;
        A(j, 2) = 1;
        A(j, 3) = pos1(match(i, 1), 1);
        b(j, 1) = pos2(match(i, 2), 1);
    end
    
    % constraints
    A(n*2+1, 3) = 1;
    b(n*2+1, 1) = 1;
    
    
    % left 
    trans_matrix = floor(A\b);
    trans_matrix(3) = [];
end
