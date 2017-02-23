function output = warpCylindrical(input, focal_length)
    output = zeros(size(input), 'uint8');
    s = focal_length;   % give less distortion
    for x = 1:size(input, 2)
        for y = 1:size(input, 1)
            x_trans = x - round(size(input, 2) / 2);
            y_trans = y - round(size(input, 1) / 2);
            theta = atan(x_trans / focal_length);
            h = y_trans / sqrt(x_trans^2 + focal_length^2);
            
            x_new = round(s * theta) + round(size(input, 2) / 2);
            y_new = round(s * h) + round(size(input, 1) / 2);

            output(y_new, x_new, :) = input(y, x, :);
        end
    end
end