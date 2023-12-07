clear;

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
    pre_r = pre(:,:,3);
    pre_r(pre_r==0) = nan;
    pre_fs(ppta) = nanmean(pre_r,'all');
    
    
    post1 = load([sample_dir,'\',post_files(ppta).name]);
    post = post1.post;
    post_r = post(:,:,3);
    post_r(post_r==0) = nan;
    post_fs(ppta) = nanmean(post_r,'all');
end
t2=clock;
t=etime(t2,t1);

TD = TD_cal(pre_fs,post_fs)
JM_dis = JM_cal(pre_fs,post_fs)

fp = fopen([fileparts(pwd),'\result.txt'],'a');
fprintf(fp,'%s\n','B');
fprintf(fp,'%s\n',['TD：',num2str(TD)]);
fprintf(fp,'%s\n',['JM：',num2str(JM_dis)]);
fprintf(fp,'%s\n\n',['Time：',num2str(t)]);
fclose(fp);

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