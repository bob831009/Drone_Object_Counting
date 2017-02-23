function [feature_x, feature_y] = feature_detection(I)
    threshold = 3;
    k = 0.04;
    plot = 0;
    boundary_wid = 6;
    edge_thres = ((10+1)^2)/10;
    focal_length = 800;

    
    [feature_x, feature_y] = feature_Harris(I, threshold, k);
   
    [feature_x, feature_y] = reject_boundary(I, feature_x, feature_y, boundary_wid);
    
    [feature_x, feature_y] = rejectLowContrast(I, feature_x, feature_y, 20);
    
    [feature_x, feature_y] = reject_edge(I, feature_x, feature_y, edge_thres);
    if(plot == 1)
        im = I;
        for i=1:length(feature_x)
            im(feature_y(i), feature_x(i)) = 255;
        end
        imshow(im);
    end
%     show image result
    
end