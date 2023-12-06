% Description: function to perform k-fold cross validation
% algorithm

function [test_data,train_data] = KFoldCrossValidation(data,fold_size)

  % code which is used to shuffle all the rows of the data set
  sort_array = randperm(size(data,1));
  for i = 1: size(data,1)
      randomized_data(i,:) = data(sort_array(i),:);
  end
  % code to divide the dataset int k sub data sets.
  no_of_rows = size(data,1);

  % initialising the test and train data with 10 (fold_size) rows for the
  % times we want
  test_data{fold_size,1} = [];
  train_data{fold_size,1} = [];

  % creating the block, which is number of pairs we want 
  block = floor(no_of_rows/fold_size);

  %test_data{1} = randomized_data(1:block,:);
  %train_data{1} = randomized_data(block+1:end,:);
  count = 1;
  % initialising array for elements we will transfer from test to train
  % data
  master_list = [];

  for f = 1:fold_size,f
      itter = 1;
      sm_list = [];
      temp_test = [];
      test_subdata = [];

      while size(test_subdata,1) <= block && count <= size(randomized_data,1)
          if ~ismember(count,master_list),itter
              temp_test = randomized_data(count,:);
              test_subdata{itter,1} = temp_test{1,1};
              test_subdata{itter,2} = temp_test{1,2};
              itter = itter + 1;
              master_list = [master_list,count];
              sm_list = [sm_list,count];
              count = count + 1;
              a = temp_test;
              b = randomized_data;
              list = [];
    
            for i=1:numel(a)
                for j=1:size(b,1)
                    for k=1:size(b,2)
                        if strcmp(a{i},b{j,k})
                            list = [list,j];
                        end
                    end
                end
            end
    
            list = unique(list);
            m_list = setdiff(list, master_list);
    
            for i=1:length(m_list)
                  temp_test = randomized_data(m_list(i),:);
                  test_subdata{itter,1} = temp_test{1,1};
                  test_subdata{itter,2} = temp_test{1,2};
                  itter = itter + 1;
                  master_list = [master_list,m_list(i)];
                  sm_list = [sm_list,m_list(i)];
            end
          else, count = count + 1;
         end
      end
      %keyboard
      test_data{f} = test_subdata;
      train_list = setdiff(1:size(randomized_data,1),sm_list);

      temp_train = [];
      for i=1:length(train_list)
          temp_train{i,1} = randomized_data(train_list(i),1);
          temp_train{i,2} = randomized_data(train_list(i),2);
      end
      train_data{f} = temp_train;
  end
end