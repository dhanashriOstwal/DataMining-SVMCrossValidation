data = dlmread('C:\Users\Dhanashri\Documents\Sem II\Data Mining\Proj2\ATNT400\HandWrittenLetters.txt');
A = data;
X = data(2:end,:);
 row = size(X,2);%1014
 %fold = 5;
[uniVal,~,index] = unique(data(1,:));  
 cnt_uniVal = numel(uniVal);     %26
 rowPerFold = row / cnt_uniVal;  %39
 testFold = floor(rowPerFold / fold);   %8
accuracy = [];
[a, b] = hist(A(1,:),unique(A(1,:)));    

n = testFold;
l = 1;
count = 0;
cnt = 1;
classes = [];
% for l = 1:5
for i = 1:fold
     testFold = floor(rowPerFold / fold);   %2
     fld(i) = testFold;
 end
abs = mod(rowPerFold,fold); 
for i = 1:abs
   fld(i) = fld(i) + 1;
end



% while (l <= fold)
%      if l==fold && mod(rowPerFold,fold)>0
%         n = n - 1;
%      end
for i = 1:fold
        n = fld(i);
        
        col = [];
        test = [];
        train = [];
        
     for k = cnt:n:row
        col = [col cnt:(cnt+(n-1))];
        %count = 1;
        cnt = cnt + rowPerFold;        
    end
        col(col > 1014) = [];
        dup = A;
        B = dup(:,col);
        test = [test B];
    
    count = count + n;
    cnt = count + 1;
    l = l + 1
    
    dup(:,col) = [];
    train = [train dup];    
    gpF = test(1,:);
    group = train(1,:);
    X =  train(2:end,:);
    Y = test(2:end,:);
    model = svmtrain(group', X', '-s 0 -t 0 -g 0.002');
    [predict_label, accuracy, dec_values] = svmpredict(gpF',Y',model);
    classes = [classes (accuracy(1))];
end
%disp(classes');
disp 'Final Accuracy='
disp(mean(classes)); 
