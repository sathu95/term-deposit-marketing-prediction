load banknormalized.csv 

bank_marketing = importdata(banknormalized.csv);

%seperating predictors from result 

predictors = bank_marketing(:,1:63);

marketing_result = bank_marketing(:,64);




















