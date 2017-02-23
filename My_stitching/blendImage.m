function imout = blendImage2(im1, im2, trans)
    trans
    if trans(1) > 0
        disp('TRANSFORMATION ERROR');
        trans(1) = -(trans(1));
    end
    % define ratio of two images
    blendWidth = size(im2, 2) + trans(1);
    imout = zeros(size(im1, 1), size(im1, 2) + size(im2, 2) - blendWidth, 3);
    r1 = ones(1, size(im1, 2));
    r2 = ones(1, size(im2, 2));
    r1(1, (end-blendWidth):end) = [1:(-1/blendWidth):0];
    r2(1, 1:(blendWidth+1)) = [0:(1/blendWidth):1];
    for k = 1:3
        for i = 1:size(im1, 2)-blendWidth
            imout(:,i,k) = double(im1(:,i,k));
        end
        for i = size(im1, 2)-blendWidth+1:size(im1, 2)
            imout(:,i,k) = double(im1(:, i,k)) * r1(i) + double(im2(:,i-size(im1, 2)+blendWidth,k)) * r2(i-size(im1, 2)+blendWidth);
        end
        for i = (size(im1, 2)+1):(size(imout, 2))
            imout(:,i,k) = double(im2(:,i-size(im1, 2)+blendWidth,k));
        end
    end
    imout(find(imout>255)) = 255;
%     imshow(uint8(imout));
end