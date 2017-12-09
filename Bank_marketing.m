
% Load data in to a file
Preprocessed_bank_marketing_data = readtable('bank_normalized.csv');

% For reproducability of training data
rng('default'); 

% Dividing the data into training and testting in the ratio 80:20
cv = cvpartition(size(Preprocessed_bank_marketing_data,1), 'Holdout', 0.80);

% Seperating Training  and Test data
trainingdata = Preprocessed_bank_marketing_data(training(cv), :);
testingdata = Preprocessed_bank_marketing_data(test(cv), :);

% Extracting predictors and response
% Processesing the data into the right shape for training the
% model.
inputTable = trainingdata;
predictorNames = {'age', 'duration', 'campaign', 'pndays', 'pcnumber', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'admin', 'blue_collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self_employed', 'services', 'student', 'technician', 'unemployed', 'job_unknown', 'divorced', 'married', 'single', 'marital_status_unknown', 'basic_4y', 'basic_6y', 'basic_9y', 'high_school', 'illiterate', 'professional_course', 'university_degree', 'education_unknown', 'default_credit_no', 'default_credit_unknown', 'default_credit_yes', 'housing_loan_no', 'housing_loan_unknown', 'housing_loan_yes', 'personal_loan_no', 'personal_loan_unknown', 'personal_loan_yes', 'cellular', 'telephone', 'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep', 'fri', 'mon', 'thu', 'tue', 'wed', 'p_c_o_failure', 'p_c_o_nonexistent', 'p_c_o_success'};
predictors = inputTable(:, predictorNames);
response = inputTable.y;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Performing cross-validation
KFolds = 5;
cvp = cvpartition(response, 'KFold', KFolds);

% Initializing the predictions to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    
    % For logistic regression, the response values must be converted to zeros
    % and ones because the responses follow a binomial
    % distribution.
    % 1 or true = 'successful' class
    % 0 or false = 'failure' class
    % NaN - missing response.
    successClass = double(1);
    failureClass = double(0);
    
    % Computing the majority response class. If there is a NaN-prediction from
    % fitglm, converting NaN to this majority class label.
    numSuccess = sum(trainingResponse == successClass);
    numFailure = sum(trainingResponse == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    successFailureAndMissingClasses = [successClass; failureClass; missingClass];
    isMissing = isnan(trainingResponse);
    zeroOneResponse = double(ismember(trainingResponse, successClass));
    zeroOneResponse(isMissing) = NaN;
    
%   Logistic Regression    
%     % Prepare input arguments to fitglm.
%     concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
%     % Train using fitglm.
%     GeneralizedLinearModel = fitglm(...
%         concatenatedPredictorsAndResponse, ...
%         'Distribution', 'binomial', ...
%         'link', 'logit');
%     
%     % Convert predicted probabilities to predicted class labels and scores.
%     convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
%     returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
%     scoresFcn = @(p) [1-p, p];
%     predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );
%     
%     % Create the result struct with predict function
%     logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
%     validationPredictFcn = @(x) logisticRegressionPredictFcn(x);
%     
%     % Add additional fields to the result struct
%     
%     % Compute validation predictions
%     validationPredictors = predictors(cvp.test(fold), :);
%     [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
%     
%     % Store predictions in the original order
%     validationPredictions(cvp.test(fold), :) = foldPredictions;
%     validationScores(cvp.test(fold), :) = foldScores;

% Classification using Random Forest
rng(1); % For reproducibility
RandomForestClassificationMdl = TreeBagger(50,trainingPredictors,trainingResponse,'OOBPrediction','On',...
    'Method','classification');
end

% Logistic Regression Accuracy
% % Compute validation accuracy
% correctPredictions = (validationPredictions == response);
% isMissing = isnan(response);
% correctPredictions = correctPredictions(~isMissing);
% validationAccuracy = sum(correctPredictions)/length(correctPredictions);

% Random Forest out-of-bag classification error 
figure;
oobErrorBaggedEnsemble = oobError(RandomForestClassificationMdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

% Test data preparation
testingpredictorNames = {'age', 'duration', 'campaign', 'pndays', 'pcnumber', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'admin', 'blue_collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self_employed', 'services', 'student', 'technician', 'unemployed', 'job_unknown', 'divorced', 'married', 'single', 'marital_status_unknown', 'basic_4y', 'basic_6y', 'basic_9y', 'high_school', 'illiterate', 'professional_course', 'university_degree', 'education_unknown', 'default_credit_no', 'default_credit_unknown', 'default_credit_yes', 'housing_loan_no', 'housing_loan_unknown', 'housing_loan_yes', 'personal_loan_no', 'personal_loan_unknown', 'personal_loan_yes', 'cellular', 'telephone', 'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep', 'fri', 'mon', 'thu', 'tue', 'wed', 'p_c_o_failure', 'p_c_o_nonexistent', 'p_c_o_success'};
testingpredictors = testingdata(:, testingpredictorNames);
testingResponse = testingdata.y;

% Confusion Matrix
[RFPredictedLabel,RFPosterior,RFCost] = predict(RandomForestClassificationMdl,testingpredictors); 
RFPredictedLabel = cell2mat(RFPredictedLabel);
RFPredictedLabel = double(RFPredictedLabel);
RFConfusionMattest = confusionmat(testingResponse,RFPredictedLabel);
RFConfusionMattest = RFConfusionMattest(1:2,3:4);
RandomForestPredictionAccuracy = RFConfusionMattest(1,1)/(RFConfusionMattest(1,1)+RFConfusionMattest(1,2));

% Plotting the Confusion Matrix
figure;
heatmap(RFConfusionMattest)
title('Random Forest Prediction Confusion Matrix');
xlabel('predicted outputs');
ylabel('actuals');


             