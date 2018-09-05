%% Filter data
% clear all;
% load('.\sparse_deepchamber.mat');
% addpath('.\admm\');
% lambda = 0.25;
% Bugres = ADMMPositionSmoothing(Res,lambda);

%% load data
load('.\sparsedeep_filtered.mat');
Bugs=Bugres.Bugs;

%% change to frame first indexing

imgpath ='.\sparse_deep\';
filelist=dir([imgpath  '*.tif']);
NumberOfFiles=length(filelist);
nI=NaN(NumberOfFiles,1);
for i=1:NumberOfFiles
    Iinf=imfinfo([filelist(i).folder filesep filelist(i).name]);
    nI(i)=length(Iinf); %this does not work for files created by ImageJ bigger than 2GB.
end
numframe=sum(nI);

Frame_first = cell(numframe,1);

for i=1:length(Bugs)
    [r,c]=size(Bugs{i,1});
    for j = 1:r
        framenum = Bugs{i,1}(j,1);
        Frame_first{framenum,1}=[Frame_first{framenum,1};Bugs{i,1}(j,2:4)];
    end
end

Filecounter = 1;


%% Process data and save image
data_count=1;
data_compiler=struct('imgpath',{},'zdepth',{});
while Filecounter < length(filelist)
    filelens = nI(Filecounter);
    
    disp(Filecounter)
    img = imread(['.\sparse_deep\',filelist(Filecounter).name],1);
    [r,c]=size(img);
    for i=1:filelens   
        framenum=sum(nI(1:Filecounter))-nI(1)+i;
        for bugnum=1:size(Frame_first{framenum,1},1)
            img_crop=cropimage(img,Frame_first{framenum,1}(bugnum,1),Frame_first{framenum,1}(bugnum,2));
            imgname = [pwd,'\imgclass_trainingset\imgclass',num2str(data_count),'.png'];
%             imwrite(img_crop,imgname);
            data_compiler(data_count).imgpath=imgname;
            bugsnear = bugsvicinity(Frame_first{framenum,1},Frame_first{framenum,1}(bugnum,:),r,c);
            data_compiler(data_count).zdepth=Frame_first{framenum,1}(bugnum,3);
            label=createlabel(bugsnear);
            data_compiler(data_count).label=label;
            data_count=data_count+1;
        end        
    end 
    Filecounter = Filecounter+1;
end
save('img_class_data_multlabel.mat','data_compiler');

%nearby bugs in the frame. [radial distance from center, zposition]
function bugsnear = bugsvicinity(buglist,position,r,c)
    center_x=position(1);
    center_y=position(2);
    
    x_idx_st=max([center_x-128,1]);
    x_idx_ed=min([center_x+127,c]);
    y_idx_st=max([center_y-128,1]);
    y_idx_ed=min([center_y+127,r]);
    nearby=(buglist(:,1)>=x_idx_st & buglist(:,1)<=x_idx_ed)&(buglist(:,2)>=y_idx_st & buglist(:,2)<=y_idx_ed);
    bugsnear = buglist(nearby,:);
    bugsnear=[((bugsnear(:,1)-center_x).^2+(bugsnear(:,2)-center_y).^2).^0.5,bugsnear(:,3)];
end

%several hot label with varying amplitudes according to the radial
%distances
function label = createlabel(bugsnear)
    label = zeros(1,2000);
    rmax=sqrt(2*128^2);
    for i=1:size(bugsnear,1)
       rad=bugsnear(i,1);
       zpos=bugsnear(i,2);
       amplitude=(1-rad/rmax);
       label(zpos)=amplitude;
    end
end

function img_crop=cropimage(img,center_x,center_y)
%crop and make the image uint8
    [r,c]=size(img);
    x_idx_st=max([center_x-128,1]);
    x_idx_ed=min([center_x+127,c]);
    y_idx_st=max([center_y-128,1]);
    y_idx_ed=min([center_y+127,r]);
    img_crop=img(y_idx_st:y_idx_ed,x_idx_st:x_idx_ed);
    
    idxingprob=[center_y-128<1, center_x-128<1, center_y+127>r, center_x+127>c];
    idxingprob = mat2str(idxingprob);
    avg=mean2(img_crop);
    padding = uint16(avg*ones(256,256));
    switch idxingprob
        case '[false false false false]'
            padding = img_crop;
        case '[true false false false]'
            padding(2-(center_y-128):256,1:256)=img_crop;
        case '[true true false false]'
            padding(2-(center_y-128):256,2-(center_x-128):256)=img_crop;
        case '[true false false true]'
            padding(2-(center_y-128):256,1:256-(center_x+127-c))=img_crop;
        case '[false true false false]'
            padding(1:256,2-(center_x-128):256)=img_crop;
        case '[false true true false]'
            padding(1:256-(center_y+127-r),2-(center_x-128):256)=img_crop;
        case '[false false true false]'
            padding(1:256-(center_y+127-r),1:256)=img_crop;
        case '[false false true true]'
            padding(1:256-(center_y+127-r),1:256-(center_x+127-c))=img_crop;
        case '[false false false true]'
            padding(1:256,1:256-(center_x+127-c))=img_crop;
    end
    img_crop=im2uint8(padding);
    
end
