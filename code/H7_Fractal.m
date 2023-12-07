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
   pre_gray = double(rgb2gray(uint8(pre)));
   pre_gray(pre_gray==0) = nan;
   pre_fs(ppta) = Simple_DBC(pre_gray);
   
   post1 = load([sample_dir,'\',post_files(ppta).name]);
   post = post1.post;
   post_gray = double(rgb2gray(uint8(post)));
   post_gray(post_gray==0) = nan;
   post_fs(ppta) = Simple_DBC(post_gray);
   
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
fprintf(fp,'%s\n','Fractal Dimention');
fprintf(fp,'%s\n',['TD：',num2str(TD)]);
fprintf(fp,'%s\n',['JM：',num2str(JM_dis)]);
fprintf(fp,'%s\n\n',['Time：',num2str(t)]);
fclose(fp);


%% calculate Fractal Dimension
function FD = Simple_DBC(raster)

step = 16;
[m1, n1]=size(raster);
if (m1<step || n1<step)
    FD = nan;
else
    
    raster=imresize(raster,[floor(m1/step)*step,floor(n1/step)*step],'nearest');
    [n1,n2] = size(raster);
    
    s = 2.^[1:log2(step)];
    h = 2.^[1:log2(step)];
    Grid_num1 = n1 ./ s;
    Grid_num2 = n2 ./ s;
    Nr = zeros(1,length(s));


    for j = 1:length(s)
        L1 =  s(j)*ones(1,Grid_num1(j));
        L2 =  s(j)*ones(1,Grid_num2(j));
        Nr(j) = nansum(nansum( cellfun(@(x) ceil(max(x(:))/h(j))-ceil(min(x(:))/h(j)) + 1,...
            mat2cell(raster,L1,L2)) ));
    end
    y = log(Nr);
    x = log(1 ./ s);
    p = polyfit(x,y,1);
    FD = abs(p(1));
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