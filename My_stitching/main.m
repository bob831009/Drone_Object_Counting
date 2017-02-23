function [total_match] = main(imdir, start_num , im_num)
    total_pos = {};
    total_desc = {};
    focal_length = 400;
    total_I = {};
    Cache = 0;
    im_scale = 1;
    

    for i = start_num:im_num
        if(i <= 10) im_index = strcat('000',num2str(i - 1));
        else im_index = strcat('00',num2str(i - 1));
        end
        impath = strcat('./', imdir, '/', im_index, '.jpg');
        I = imresize(rgb2gray(imread(impath)), im_scale);
        I = warpCylindrical(I, focal_length);
        total_I{i} = warpCylindrical(imresize(imread(impath), im_scale), focal_length);
        
        [feature_x, feature_y] = feature_detection(I);
        if(Cache)
            load(sprintf('./mat/pos_%02d.mat', i));
            load(sprintf('./mat/desc_%02d.mat', i));
        else
            
            [pos, desc] = feature_description(imresize(imread(impath), im_scale), feature_x, feature_y);
            save(sprintf('./mat/pos_%02d.mat', i), 'pos');
            save(sprintf('./mat/desc_%02d.mat', i), 'desc');
        end
        total_pos{i} = pos;
        total_desc{i} = desc;
    end
    
    total_match = {};
    for i = start_num:im_num-1
%         length(total_desc{i})
        if(Cache)
            load(sprintf('./mat/match_%02d.mat', i));
        else
            [match] = feature_matching(total_desc{i}, total_desc{i+1}, total_pos{i}, total_pos{i+1});
            match_len = length(match)
            save(sprintf('./mat/match_%02d.mat', i), 'match');
        end
        match = Ransac(match, total_pos{i}, total_pos{i+1});
        total_match{i} = match;
        
    end
    
    trans_matrix = {};
    for i = start_num:im_num-1
        [output] = imageMatching(total_match{i}, total_pos{i}, total_pos{i+1});
        trans_matrix{i} = output;
    end
    
    
    imNow = total_I{start_num};
    for i = start_num:im_num-1
        imNow = blendImage(imNow, total_I{i+1}, trans_matrix{i});
    end
    for i= size(imNow, 1):-1:1
        if(sum(sum(imNow(i,:,:))) == 0)
            imNow(i, :, :) = [];
        end
    end
%     imshow(uint8(imNow));
    imwrite(uint8(imNow), strcat(imdir, '_result.jpg'));
end