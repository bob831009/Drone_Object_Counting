function [feature_x, feature_y] = rejectLowContrast(I, feature_x, feature_y, threshold)
    
    im = I;
	threshold = 20; % 15/255
    [row, col] = size(I);
    I = double(I);

    k = fspecial('average', 5);
    contrast = abs(filter2(k, I, 'same') - I);

    newX = [];
    newY = [];
    for i = 1:numel(feature_x)
	if( contrast(feature_y(i), feature_x(i)) > threshold )
	    newY = [newY feature_y(i)];
	    newX = [newX feature_x(i)];
	end
    end

    feature_y = newY;
    feature_x = newX;
end