function [result] = Gaussianfilter(I, sigma, w)
    filter = fspecial('gaussian', [w w], sigma);
    result = imfilter(I, filter, 'same');
end