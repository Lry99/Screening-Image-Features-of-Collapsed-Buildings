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
   pre_hsi = rgb2hsi(pre);
   pre_I = pre_hsi(:,:,3);
   pre_valid = pre_I(pre_I~=0);
   pre_fs(ppta) = mean(pre_valid);


   post1 = load([sample_dir,'\',post_files(ppta).name]);
   post = post1.post;
   post_hsi = rgb2hsi(post);
   post_I = post_hsi(:,:,3);
   post_valid = post_I(post_I~=0);
   post_fs(ppta) = mean(post_valid);
end
t2=clock;
t=etime(t2,t1);

TD = TD_cal(pre_fs,post_fs)
JM_dis = JM_cal(pre_fs,post_fs)

fp = fopen([fileparts(pwd),'\result.txt'],'a');
fprintf(fp,'%s\n','I');
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

%% Color space transformation
function [hsi] = rgb2hsi(rgb)

rgb=double(rgb);
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);

%   H
num=0.5*((r-g)+(r-b));
den=sqrt( (r-g).^2 + (r-b).*(g-b) );
theta=acos(num./(den+eps));
H0=theta.*(g>=b);
H1=(2*pi-theta).*(g<b);
H=H0+H1;


%   S
num=3.*min(min(r,g),b);
S=1-num./(r+g+b+eps);

%   I
I=(r+g+b)/3;

H=(H-min(min(H)))./(max(max(H))-min(min(H)));
S=(S-min(min(S)))./(max(max(S))-min(min(S)));

hsi=cat(3,H,S,I);

end