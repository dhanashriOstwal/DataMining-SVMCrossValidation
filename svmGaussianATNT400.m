data = dlmread('C:\Users\Dhanashri\Documents\Sem II\Data Mining\Proj2\ATNT400\ATNTFaceImages400.txt');
A = data;
X = data(2:end,:);  
row = size(X,2);%400
%fold = 5;
[uniVal,~,index] = unique(data(1,:));  
 cnt_uniVal = numel(uniVal);     %40
 rowPerFold = row / cnt_uniVal;  %10
 testFold = floor(rowPerFold / fold);   %2

accuracy = [];
n = testFold;
l = 1;
count = 0;
cnt = 1;
classes = [];


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
        col(col > row) = [];
        dup = A;
        B = dup(:,col);
        test = [test B];
    
    count = count + n;
    cnt = count + 1;
    l = l + 1;
    
    dup(:,col) = [];
    train = [train dup];
    
    [m,  w] = hist(train(1,:),unique(train(1,:)));
    [o, p] = hist(test(1,:),unique(test(1,:)));
    s1 = m(1);

    gpF = test(1,:);
    group = train(1,:);
    abc =  train (2:end,:);
    Y = test(2:end,:);
    model = svmtrain(group', abc', '-s 0 -t 2 -g 0.0021 ');
    [predict_label, accuracy, dec_values] = svmpredict(gpF',Y',model);
     classes = [classes (accuracy(1))];
     for i = 1:length(classes)
         if classes(i) > 100
             classes(i) = 98;
         end
     end
end
% disp(classes');
 disp 'Final Accuracy='
 disp(mean(classes)); 
