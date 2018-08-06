load('img_class_data.mat')
% b={'D:','Jong','imageclassification','imgclass_train_jpeg'};
b={'..','imgclass_png'};
for i=1:length(data_compiler)
   a=data_compiler(i).imgpath;
   a=strsplit(a,'\');
   file=a{7};
   file=strsplit(file,'s');
   file=strsplit(file{2},'.');
   b{3}=['imgclass_Z',int2str(i-1),'.png'];
   modified=strjoin(b,'\');
   data_compiler(i).imgpath=modified;
end

save('img_class_png.mat','data_compiler')