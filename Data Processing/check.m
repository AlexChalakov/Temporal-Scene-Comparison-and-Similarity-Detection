c = [];

for k =1:10
    a = test_data{k};
    b = train_data{k};

    for i=1:numel(a)
        for j=1:numel(b)
            if strcmp(a{i},b{j}) == 1
                i2 = i;
                j2 = j;
                if i2 > size(a,1),i2 = i2 - size(a,1);end
                if j2 > size(b,1),j2 = j2 - size(b,1);end
                c = [c;k,i,j,i2,j2];
            end
        end
    end
end

x = 10;
length(unique(c(c(:,1)==x,4)))
length(unique(c(c(:,1)==x,5)))
%note down - table for report
%determing overlap between testing and training
%remove overlaps
%explain consistency about removing from testinunique(c(c(:,1)==x,4)))
unique(c(c(:,1)==x,4))
%tells us rows