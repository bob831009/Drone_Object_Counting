function [match] = feature_matching(desc1, desc2, pos1, pos2)
    match = [];
    for i = 1:length(desc1)
        feature_point1 = desc1{i};
        distance = [];
        for j = 1:length(desc2)
            feature_point2 = desc2{j};
            tmp_dis = sqrt(sum((feature_point1 - feature_point2).^2));
            distance = [distance; tmp_dis];
        end
        
        [min1, min1_index] = min(distance);
        distance(min1_index) = [];
        [min2, min2_index] = min(distance);
        
        
        if(min1/min2 < 0.85)
            match = [match; [i, min1_index]];
        end
%         [min1 , min2]
    end
end