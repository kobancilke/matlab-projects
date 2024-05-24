clear all;
clc;

% Load the BNT library
addpath(genpath('C:\Users\ikoba\Desktop\PWr_Courses\Artificial Ingelligent and Machine Learning\BayesianBeliefNetworks\bnt-master'));  % Path to the BNT
import bnt.*;
import bnt.BNT.*;

% Load and prepare the training dataset
data = readtable('loan_data.csv');

% Convert categorical data to numeric
data.Gender = grp2idx(data.Gender);
data.Married = grp2idx(data.Married);
data.Dependents = grp2idx(data.Dependents);
data.Education = grp2idx(data.Education);
data.Self_Employed = grp2idx(data.Self_Employed);
data.Property_Area = grp2idx(data.Property_Area);
data.Loan_Status = grp2idx(data.Loan_Status);

% Define the number of variables
N = width(data);  % Adjust according to the actual number of columns

% Define node indices
Gender = 1; Married = 2; Dependents = 3; Education = 4; Self_Employed = 5;
Applicant_Income = 6; Coapplicant_Income = 7; Credit_History = 8; Property_Area = 9; Loan_Status = 10;

% Define the DAG structure
dag = zeros(N, N);
dag([Gender, Married, Dependents, Education, Self_Employed, Property_Area], [Applicant_Income, Coapplicant_Income]) = 1;
dag([Applicant_Income, Coapplicant_Income, Credit_History], Loan_Status) = 1;

% Define probability distributions
node_sizes = 2 * ones(1, N);  % Assuming binary nodes for simplicity, adjust if needed
bnet = mk_bnet(dag, node_sizes, 'discrete', 1:N);

% Initialize CPDs to uniform distributions
for i = 1:N
    bnet.CPD{i} = tabular_CPD(bnet, i);
end

% Convert table to matrix and prepare cases
data_matrix = table2array(data);
cases = cell(N, size(data_matrix, 1));
for i = 1:size(data_matrix, 1)
    cases(:, i) = num2cell(data_matrix(i, :));
end

% Create an inference engine
engine = jtree_inf_engine(bnet);

% Learn the CPDs using the prepared data
try
    bnet_learned = learn_params(bnet, cases);
    disp('Model training successful.');
catch ME
    disp('Error during model training:');
    disp(ME.message);
end

% Step 3 - Calculating Probabilities
evidence = cell(1, N);
evidence{Credit_History} = 1;  % Known credit history

[engine, ll] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, Loan_Status);

% Display the calculated probabilities with explanations
disp('P(Loan_Status | Credit_History = 1):');
disp(['P(Loan_Status = 1 | Credit_History = 1): ', num2str(marg.T(2)), ' (Probability of loan being approved)']);
disp(['P(Loan_Status = 0 | Credit_History = 1): ', num2str(marg.T(1)), ' (Probability of loan being rejected)']);

% Step 4 - Decision Making
% Adding the decision node and utility node
decision_node = N + 1;  % Added as a new node
utility_node = N + 2;   % Utility node

% Define the new DAG structure
dag = zeros(N + 2, N + 2);
dag([Gender, Married, Dependents, Education, Self_Employed, Property_Area], [Applicant_Income, Coapplicant_Income]) = 1;
dag([Applicant_Income, Coapplicant_Income, Credit_History], Loan_Status) = 1;
dag(decision_node, Loan_Status) = 1;  % Connection from decision node to Loan_Status
dag([Loan_Status, decision_node], utility_node) = 1;  % Connections to utility node

% Define the new BBN
node_sizes = [node_sizes 2 1];  % Adding decision node and utility node
bnet = mk_bnet(dag, node_sizes, 'discrete', 1:N + 1);

% Redefine CPDs
for i = 1:N
    bnet.CPD{i} = tabular_CPD(bnet, i);
end
bnet.CPD{decision_node} = tabular_CPD(bnet, decision_node);

% Define the utility function
utility_function = @(loan_status) (loan_status == 1) * 100 - (loan_status == 0) * 10;

% Calculate expected utility with evidence known
prob_loan_approved_given_evidence = marg.T(2);  % Probability of loan being approved given evidence
prob_loan_rejected_given_evidence = marg.T(1);  % Probability of loan being rejected given evidence

EU_given_evidence = prob_loan_approved_given_evidence * utility_function(1) + ...
                    prob_loan_rejected_given_evidence * utility_function(0);

% Calculate expected utility without any evidence (prior probability)
% Here, you will need the prior probabilities or a way to estimate them
% For demonstration, assume prior probabilities as uniform or based on data
prior_prob_loan_approved = 0.5;  % This should be based on actual data or model estimation
prior_prob_loan_rejected = 0.5;  % This should be based on actual data or model estimation

EU_without_evidence = prior_prob_loan_approved * utility_function(1) + ...
                      prior_prob_loan_rejected * utility_function(0);

% Calculate Value of Perfect Information
VPI = EU_given_evidence - EU_without_evidence;

% Display VPI
disp(['Value of Perfect Information: ', num2str(VPI)]);