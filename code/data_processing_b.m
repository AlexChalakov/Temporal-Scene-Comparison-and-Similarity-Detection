function data_processing_b()

idx_fn = 'Medium/Medium_indexes.txt';

tab = readtable(idx_fn,'delimiter',' ','ReadVariableNames',false);

ntab = table;
ntab.p1 = cell(height(tab),1);
ntab.p2 = cell(height(tab),1);
% go through table and replace with filename
for i = 1:height(tab)
    p1 = tab{i,1}{1};
    p2 = tab{i,2}{1};
    s1 = strfind(p1,'\');
    s2 = strfind(p2,'\');
    p1 = p1(s1(end)+1:end);
    p2 = p2(s2(end)+1:end);
    ntab.p1{i,1} = (p1);
    ntab.p2{i,1} = (p2);
end

% start to consolidate
combcell = unique([ntab.p1;ntab.p2]);
indx = (1:length(combcell))';

% make matrix of table to make life easier
vma = zeros(height(ntab),2);
for i = 1:height(ntab)
    vma(i,1) = find(strcmp(ntab.p1{i,1},combcell));
    vma(i,2) = find(strcmp(ntab.p2{i,1},combcell));
end

% start clustering
noerrs = inf;
ite = 0;
while noerrs > 0
    ite = ite + 1,
    noerrs = 0;
    ul = unique(indx);
    nindx = indx;
    for i = 1:length(ul)
        tul = ul(i);
        indlst = find(indx==tul);
        imlst = indlst;
        % for each image, do a check to see what it's linked to
        for j = 1:length(imlst)
            [tind,~] = find(vma==imlst(j));
            tind = unique(tind);
            newimlst = unique([vma(tind,1);vma(tind,2)]);
            for k = 1:length(newimlst)
                if nindx(newimlst(k)) ~= tul
                    nindx(newimlst(k)) = tul;
                    noerrs = noerrs + 1;
                end
            end
        end
    end
    indx = nindx;
end

% relabel the index
uin = unique(indx);
for i = 1:length(uin)
    indx(indx==uin(i)) = i;
end

% Make the new table with some validations specified
% specify validations (20% data) and test (20% data)
rp = randperm(max(indx));

ntab2 = ntab;
for i = 1:height(ntab)
    lab = indx(vma(i,1));
    rploc = find(rp==lab) / length(rp);
    if rploc <= 0.6 % then training
        famname = ['0000' num2str(indx(vma(i,1)))]; famname = ['TrF' famname(end-3:end)];
    elseif rploc <= 0.8 % validation
        famname = ['0000' num2str(indx(vma(i,1)))]; famname = ['VaF' famname(end-3:end)];
    else % testing
        famname = ['0000' num2str(indx(vma(i,1)))]; famname = ['TeF' famname(end-3:end)];
    end
    imnm1 = ntab.p1{i,1};
    imnm2 = ntab.p2{i,1};
    ntab2.p1{i,1} = [famname '/' imnm1];
    ntab2.p2{i,1} = [famname '/' imnm2];
end

% dump ext for directory to mimic structure
for i = 1:height(ntab2)
    ntab2.p1{i,1} = ntab2.p1{i,1}(1:end-4);
    ntab2.p2{i,1} = ntab2.p2{i,1}(1:end-4);
end
writetable(ntab2,'Medium/Medium_Relationships.csv','delimiter',',');

%% To make the composite folder
inp = 'Medium/images';
oup = 'Medium/images-structured';
for i = 1:height(ntab)
    p1i = [inp '/' ntab.p1{i,1}];
    mkdir([oup '/' ntab2.p1{i,1}]);
    p1o = [oup '/' ntab2.p1{i,1} '/' ntab.p1{i,1}];
    copyfile(p1i,p1o);
    p2i = [inp '/' ntab.p2{i,1}];
    mkdir([oup '/' ntab2.p2{i,1}]);
    p2o = [oup '/' ntab2.p2{i,1} '/' ntab.p2{i,1}];
    copyfile(p2i,p2o);
end
