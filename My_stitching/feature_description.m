function [pos, desc] = feature_description(I, feature_x, feature_y)
    I = double(I);
    pos = [];
    for i = 1:length(feature_x)
        x = feature_x(i);
        y = feature_y(i);
        pos = [pos; [y, x]];
        tmp_v = reshape(I(y-6:y+6, x-6:x+6, :), 1, []);
        desc{i} = tmp_v;
    end
end