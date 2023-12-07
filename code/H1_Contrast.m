clear;

s_wins = [3 5 7 9];
levels = [4 8 16 32 64 128 256];

for s_win = s_wins
    for level = levels

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
            pre_fs(ppta) = m_GLCM_Constract(pre_gray,s_win,level);


            post1 = load([sample_dir,'\',post_files(ppta).name]);
            post = post1.post;
            post_gray = double(rgb2gray(uint8(post)));
            post_gray(post_gray==0) = nan;
            post_fs(ppta) = m_GLCM_Constract(post_gray,s_win,level);
            
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
        fprintf(fp,'%s\n','Contrast');
        fprintf(fp,'%s\n',['window size：',num2str(s_win)]);
        fprintf(fp,'%s\n',['level：',num2str(level)]);
        fprintf(fp,'%s\n',['TD：',num2str(TD)]);
        fprintf(fp,'%s\n',['JM：',num2str(JM_dis)]);
        fprintf(fp,'%s\n\n',['Time：',num2str(t)]);
        fclose(fp);


    end
end




%% calculate GLCM of a smaple
function mean_GLCM = m_GLCM_Constract(raster,s_win,level)

[n1,n2] = size(raster);
neighbour_GLCM = zeros(n1,n2);
half = (s_win-1)/2;

for i = half+1:n1-half
    for j = half+1:n2-half-1
        windw = raster(i-half:i+half,j-half:j+half+1);
        if isempty(find(isnan(windw), 1))
            neighbour_GLCM(i,j) = GLCM_Contrast(windw,level);
        else
            neighbour_GLCM(i,j) = nan;
        end
    end
end
mean_GLCM = nanmean(neighbour_GLCM(half+1:n1-half,half+1:n2-half-1),'all');
end

function a = GLCM_Contrast(win,bin)

GLCM = graycomatrix(win,'NumLevels',bin,'GrayLimits',[0 256]);
stats = graycoprops(GLCM);

a = stats.Contrast;
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