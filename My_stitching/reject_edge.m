function [feature_x, feature_y] = reject_edge(I, feature_x, feature_y, threshold)
    
    [row, col] = size(I);
    I = double(I);
    new_x = [];
    new_y = [];
    
    for i=1:length(feature_x)
        x = feature_x(i);
        y = feature_y(i);
        
        Dxx = I(y, x-1) -2*I(y, x) + I(y, x+1);
        Dyy = I(y-1, x) -2*I(y, x) + I(y+1, x);
        Dxy = (I(y-1, x-1) -I(y-1, x+1) -I(y+1, x-1) +I(y+1, x+1))/4;
        
        
        TrH = Dxx + Dyy;
        DetH = Dxx*Dyy - Dxy^2;
        
        if( (TrH^2) / DetH < threshold & DetH > 0 )
            new_x = [new_x, x];
            new_y = [new_y, y];
        end
    end
    
    feature_x = new_x;
    feature_y = new_y;
end