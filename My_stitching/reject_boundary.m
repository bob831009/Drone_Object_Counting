function [feature_x, feature_y] = reject_boundary(I, x, y, bound_wid)
    
    [row, col] = size(I);
    
    new_x = [];
    new_y = [];
    for i = 1:length(x)
        if( (x(i) > bound_wid && x(i) < col-bound_wid) && (y(i) > bound_wid && y(i) < row - bound_wid))
            new_x = [new_x, x(i)];
            new_y = [new_y, y(i)];
        end
    end
    
    feature_x = new_x;
    feature_y = new_y;
end