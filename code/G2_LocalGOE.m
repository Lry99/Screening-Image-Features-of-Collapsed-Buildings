clear;

s_wins = [3 5 7 9];
orientations = [4 8 16 18 45];
for s_win = s_wins
    for orientation = orientations

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
            pre_fs(ppta) = sobel_gradient_DirC(pre_gray,s_win,orientation);

            post1 = load([sample_dir,'\',post_files(ppta).name]);
            post = post1.post;
            post_gray = double(rgb2gray(uint8(post)));
            post_gray(post_gray==0) = nan;
            post_fs(ppta) = sobel_gradient_DirC(post_gray,s_win,orientation);

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
        fprintf(fp,'%s\n','LocalGDE');
        fprintf(fp,'%s\n',['window size：',num2str(s_win)]);
        fprintf(fp,'%s\n',['orientation：',num2str(orientation)]);
        fprintf(fp,'%s\n',['TD：',num2str(TD)]);
        fprintf(fp,'%s\n',['JM：',num2str(JM_dis)]);
        fprintf(fp,'%s\n\n',['Time：',num2str(t)]);
        fclose(fp);
    end
end

%% calculate Local Gradient Orientation Entropy of a sample
function result = sobel_gradient_DirC(raster,s_win,orientation)

hx = fspecial('prewitt');
hy = transpose(hx);

gradx = imfilter(raster,hx,'same');
grady = imfilter(raster,hy,'same');

Direction = atan(grady./gradx)/pi*180 + 90;
Direction = Direction(2:end-1,2:end-1);

Direction_num = floor(Direction/(180/orientation));
Direction_num(Direction_num == max(Direction_num(:))) = 0;

[n1,n2] = size(Direction_num);
Direction_neighbour_C = zeros(n1,n2);
half = (s_win-1)/2;

for i = half+1:n1-half
    for j = half+1:n2-half
        windw = Direction_num(i-half:i+half,j-half:j+half);
        if isempty(find(isnan(windw), 1))
            Direction_neighbour_C(i,j) = comentropy(windw,orientation);
        else
            Direction_neighbour_C(i,j) = nan;
        end
    end
end
result = nanmean(Direction_neighbour_C(half+1:n1-half,half+1:n2-half),'all');
end

function result = comentropy(A,level)

sizeA = size(A,1)*size(A,2);
temp=zeros(1,level);
for pTa = 1:sizeA
    i = A(pTa)+1;
    temp(i) = temp(i)+1;
end
temp = temp/sizeA;

result=0;
for tTa=1:length(temp)
    if temp(tTa)~=0
        result=result-temp(tTa)*log2(temp(tTa));
    end
end

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