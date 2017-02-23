function max_match_pair = Ransac(match, pos1, pos2)
    p = 0.5;
    n = 2;
    P = 0.9999;
    threshold = length(match)*0.5;
	total_test_time = ceil(log(1-P)/log(1-p^n));
    N = length(match);

    max_match_pair = [];
    if N <= n
        max_match_pair = match;
        return;
    end
    
    for test_time = 1:total_test_time
        pool = randperm(N);
        index_now = pool(1:n);

        Match_now = match(index_now, :);
        otherMatch = match;
        otherMatch(index_now, :) = [];

        Match_now_pos1 = pos1(Match_now(:,1), :);
        Match_now_pos2 = pos2(Match_now(:,2), :);
        Match_others_pos1 = pos1(otherMatch(:,1), :);
        Match_others_pos2 = pos2(otherMatch(:,2), :);

        tmpMatch = [];
        posDiff = Match_now_pos1 - Match_now_pos2;
        theta = mean(posDiff);
        
        for i = 1:size(Match_others_pos1, 1)
            distance = (Match_others_pos1(i,:)-Match_others_pos2(i,:)) - theta;
            if sqrt(sum(distance.^2)) < threshold
                tmpMatch = [tmpMatch; otherMatch(i, :)];
            end
        end
        if size(tmpMatch, 1) > size(max_match_pair, 1)
            max_match_pair = tmpMatch;
        end
    end
end
