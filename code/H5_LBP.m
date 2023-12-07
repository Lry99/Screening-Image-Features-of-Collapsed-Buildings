clear;

radiuses = [1:2:9];

for radius = radiuses

    sample_dir = [fileparts(pwd),'\data\single_cons'];
    pre_files = dir([sample_dir,'\*pre.mat']);
    post_files = dir([sample_dir,'\*post.mat']);
    len_sample = length(pre_files);

    pre_fs = zeros(len_sample,1);
    post_fs = zeros(len_sample,1);


    t1=clock;
    parfor ppta = 1:len_sample

        pre1=load([sample_dir,'\',pre_files(ppta).name]);
        pre = pre1.pre;
        pre_gray = double(rgb2gray(uint8(pre)));
        pre_gray(pre_gray==0) = nan;
        pre_fs(ppta) = cal_LBP(pre_gray,radius);

        post1 = load([sample_dir,'\',post_files(ppta).name]);
        post = post1.post;
        post_gray = double(rgb2gray(uint8(post)));
        post_gray(post_gray==0) = nan;
        post_fs(ppta) = cal_LBP(post_gray,radius);

    end
    t2=clock;
    t=etime(t2,t1);

    pre_fs_copy = pre_fs;
    post_fs_copy = post_fs;
    pre_NaN = isnan(pre_fs);
    post_NaN = isnan(post_fs);
    pre_fs_copy(pre_NaN) = [];
    post_fs_copy(post_NaN) = [];

    TD = TD_cal(pre_fs_copy,post_fs_copy)
    JM_dis = JM_cal(pre_fs_copy,post_fs_copy)

    fp = fopen([fileparts(pwd),'\result.txt'],'a');
    fprintf(fp,'%s\n','LBP');
    fprintf(fp,'%s\n',['radius：',num2str(radius)]);
    fprintf(fp,'%s\n',['TD：',num2str(TD)]);
    fprintf(fp,'%s\n',['JM：',num2str(JM_dis)]);
    fprintf(fp,'%s\n\n',['Time：',num2str(t)]);
    fclose(fp);
end


%% calculate LBP
function result = cal_LBP(raster,radius)

radius = radius;
neighbors = 8;
[rows, cols] = size(raster);
rows=int16(rows);
cols=int16(cols);
imglbp = uint8(zeros(rows-2*radius, cols-2*radius));


for k=0:neighbors-1

    rx = radius * cos(2.0 * pi * k / neighbors);
    ry = -radius * sin(2.0 * pi * k / neighbors);

    x1 = floor(rx); 
    x2 = ceil(rx); 
    y1 = floor(ry);
    y2 = ceil(ry); 

    tx = rx - x1;
    ty = ry - y1;

    w1 = (1-tx) * (1-ty);
    w2 = tx * (1-ty);
    w3 = (1-tx) * ty;
    w4 = tx * ty;

    for i = radius+1:rows-radius
        for j = radius+1:cols-radius
            center = raster(i, j);

            neighbor = raster(i+x1, j+y1)*w1 + raster(i+x1, j+y2)*w2 + raster(i+x2, j+y1)*w3 + raster(i+x2, j+y2)*w4;
            
            if neighbor > center
                flag = 1;
            else
                flag = 0;
            end

            
            imglbp(i-radius, j-radius) = bitor(imglbp(i-radius, j-radius), bitshift(flag, neighbors-k-1));
        end
    end
end

for i = 1:rows-2*radius
    for j = 1:cols-2*radius
        currentValue = imglbp(i, j);
        minValue = currentValue;
        currentValue = dec2bin(currentValue);
        
        for k=1:neighbors
            temp = circshift(currentValue, k);
            temp = bin2dec(temp);

            if temp < minValue
                minValue = temp;
            end
        end
        imglbp(i, j) = minValue;
    end
end
result = nanmean(imglbp,'all');


end
%% calculate TD
function TD = TD_cal(i,j)

U_i = mean(i,1);
U_j = mean(j,1);

Cov_i = cov(i);
Cov_j = cov(j);

D_ij = 0.5 * sum(diag((Cov_i-Cov_j)*(inv(Cov_j)-inv(Cov_i))))...
    + 0.5 * sum(diag((inv(Cov_i)+inv(Cov_j))*transpose(U_i-U_j)*(U_i-U_j)));
TD = 2*(1-exp(-0.125*D_ij));

end

%% calculate J-M distance
function JM_dis = JM_cal(i,j)

U_i = mean(i);
U_j = mean(j);

Cov_i = cov(i);
Cov_j = cov(j);

p = (Cov_i+Cov_j)/2;

B = 0.125 *  (U_i-U_j)* inv(p) * transpose(U_i - U_j)...
    + 0.5 * log( det(p) / ( sqrt(det(Cov_i)) * sqrt(det(Cov_j)) ) );
JM_dis = 2 * (1-exp(-B));
end