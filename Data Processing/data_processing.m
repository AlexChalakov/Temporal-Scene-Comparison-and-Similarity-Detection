%Reading the medium indexes and putting them in a table so the
%relationships are not ruined.
%We use space as a Delimiter so that each photo is split into its own cell.
ID = readtable('Dataset/Indexes/Medium_indexes.txt', 'Delimiter', 'space');

%in case an array is needed
Arr = table2array(ID);
No_of_folds = 10;
[test_data,train_data] = KFoldCrossValidation(Arr,No_of_folds);

save([mfilename '.mat']);
for i = 1:10
    %table creation
    txt = sprintf('ttr%s = table', num2str(i));
    eval(txt);
    txt = sprintf('tti%s = table', num2str(i));
    eval(txt);

    %column 1
    txt = sprintf('ttr%s.p1 = train_data{%.0f}(:,1)', num2str(i), i);
    eval(txt);
    txt = sprintf('tti%s.p1 = test_data{%.0f}(:,1)', num2str(i), i);
    eval(txt);

    %column 2
    txt = sprintf('ttr%s.p2 = train_data{%.0f}(:,2)', num2str(i), i);
    eval(txt);
    txt = sprintf('tti%s.p2 = test_data{%.0f}(:,2)', num2str(i), i);
    eval(txt);

    %tables
    txt = sprintf('writetable(ttr%s, ''train_relationships_p%s.csv'')', num2str(i), num2str(i));
    eval(txt);
    txt = sprintf('writetable(tti%s, ''test_relationships_p%s.csv'')', num2str(i), num2str(i));
    eval(txt);
end