function [feature_x, feature_y] = feature_Harris(I, threshold, k)
    sigma = 1;
    filter_width = 5;
    I = double(I);
    
    gaussI = Gaussianfilter(I, sigma, filter_width);
    [Ix, Iy] = gradient(gaussI);
    
    Ix2 = Ix .^ 2;
    Iy2 = Iy .^ 2;
    Ixy = Ix .* Iy;
    
    Sx2 = Gaussianfilter(Ix2, sigma, filter_width);
    Sy2 = Gaussianfilter(Iy2, sigma, filter_width);
    Sxy = Gaussianfilter(Ixy, sigma, filter_width);
    
    R = (Sx2 .* Sy2 - Sxy .* Sxy) - k*(Sx2 + Sy2) .^ 2;
    
    R_thres = (R > threshold);
%     compute nonmax suppression.
    R_thres = R_thres & (R > imdilate(R, [1 1 1; 1 0 1; 1 1 1]));
%     find feature_point
    [feature_y, feature_x]= find(R_thres);

end