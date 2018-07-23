load('img_class_data.mat')
b={'D:','Jong','imageclassification','imgclass_train_jpeg'};

for i=1:length(data_compiler)
   a=data_compiler(i).imgpath;
   a=strsplit(a,'\');
   file=a{7};
   file=strsplit(file,'.');
   b{5}=[file{1},'.jpg'];
   modified=strjoin(b,'\');
   data_compiler(i).imgpath=modified;
end

save('img_class_data_jpg_d.mat','data_compiler')