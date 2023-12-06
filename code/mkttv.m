function mkttv()

td = 'Medium/test-images'
mkdir("Medium/test-images");

% rename test data to make the relationships clearer
di = dir('Medium/images-structured/Te*/*/*.jpg');
for i = 1:length(di)
    sf = strfind(di(i).folder,'/');
    v1 = di(i).folder(sf(end-1)+1:sf(end)-1);
    v2 = di(i).folder(sf(end)+1:end);
    ofn = [v1 '^' v2 '^' num2str(i) '.jpg'];
    copyfile([di(i).folder '/' di(i).name],['Medium/test-images/' ofn]);
end

% make a testing index for testing images 
di = dir([td '/*.jpg']);
t = table;
t.img_pair = cell(2*length(di),1);
t.is_related = zeros(2*length(di),1);
for i = 1:length(di)
    v1 = di(i).name;
    %di2 = dir([td '/' di(i).name(1:8) '*.jpg']);
    v1s = strfind(v1,'^');
    di2 = dir([td '/' v1(1:v1s(1)-1) '*.jpg']);
    ls = randperm(length(di2));
    v2 = di2(ls(1)).name;
    if strcmp(v1,v2), v2 = di2(ls(2)).name; end
    ls = randperm(length(di));
    for k = 1:length(ls)
        v3 = di(ls(k)).name;
        v3s = strfind(v3,'^')
        if strcmp(v1(1:v1s(1)-1),v3(1:v3s(1)-1))==0, break; end
    end
    t.img_pair{2*i-1,1} = [v1 '-' v2];
    t.is_related(2*i-1,1) = 1;
    t.img_pair{2*i  ,1} = [v1 '-' v3];
end
writetable(t,'sample_submission_Medium.csv');




% 
% t = readtable("sample_submission.csv","Delimiter",",");
% c = unique([t.p1;t.p2]);
% t = table;
% t.p = c;
% writetable(t,[mfilename '.csv']);