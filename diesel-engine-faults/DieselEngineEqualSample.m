clc;
clear;

data = load('Features_2500RPM_0dB_full.mat'); 
DataBase_table = data.DataBase_table; 

% Uncomment to see the size of the database table
% disp(size(DataBase_table));
% Uncomment to see the summary of the database table
% summary(DataBase_table);

features = DataBase_table(:,1:84); % Assuming the first 84 columns are features
featuresMatrix = table2array(features); % Convert table to array for processing

group1_features = featuresMatrix(1:250, :); % Group 1
group2_features = featuresMatrix(251:500, :); % Group 2
group3_features = featuresMatrix(501:2000, :); % Group 3
group4_features = featuresMatrix(2001:3500, :); % Group 4

% Selecting random 250 rows from Group 3 and 4
rng(1); % Set seed for randomness
rand_indices3 = randperm(1500, 250);
rand_indices4 = randperm(1500, 250);
selected_group3_features = group3_features(rand_indices3, :);
selected_group4_features = group4_features(rand_indices4, :);

labels = [ones(250, 1); 2*ones(250, 1); 3*ones(250, 1); 4*ones(250, 1)];  % Labels
labels = categorical(labels, 1:4, {'NormalOperation', 'PressureReduction', 'CompressionReduction', 'FuelReduction'});

final_features = [group1_features; group2_features; selected_group3_features; selected_group4_features];
shuffled_indices = randperm(height(final_features));
shuffled_features = final_features(shuffled_indices,:);
shuffled_labels = labels(shuffled_indices);

% Ratios of train, validation, and test data
% total_samples = height(shuffled_data);
% train_ratio = 0.7;
% validation_ratio = 0.1;
% test_ratio = 0.2;

% Separate data according to train, validation, and test subsets
total_samples = height(shuffled_features);
train_samples = floor(0.7 * total_samples);
validation_samples = floor(0.1 * total_samples);
test_samples = total_samples - train_samples - validation_samples;

train_features = shuffled_features(1:train_samples, :);
train_labels = shuffled_labels(1:train_samples);

validation_features = shuffled_features(train_samples+1:train_samples+validation_samples, :);
validation_labels = shuffled_labels(train_samples + 1:train_samples + validation_samples);

test_features = shuffled_features(train_samples + validation_samples + 1:end, :);
test_labels = shuffled_labels(train_samples + validation_samples + 1:end);

% Deep CNN model configuration
layers = [
    imageInputLayer([1 84 1])
    convolution2dLayer(5, 16, 'Padding', 'same') %convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    convolution2dLayer(5, 32, 'Padding', 'same') % convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(1, 'Stride', 2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 30, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {reshape(validation_features', [1 84 1 size(validation_features, 1)]), validation_labels}, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Model Training
[net, info] = trainNetwork(reshape(train_features', [1 84 1 size(train_features, 1)]), train_labels, layers, options);

disp('Validation Step:');
predictions = classify(net, reshape(validation_features', [1 84 1 size(validation_features, 1)]));
accuracy = sum(predictions == validation_labels) / numel(validation_labels);
disp(['Validation Accuracy: ', num2str(accuracy)]);

disp('Test Step:');
predictions = classify(net, reshape(test_features', [1 84 1 size(test_features, 1)]));
accuracy = sum(predictions == test_labels) / numel(test_labels);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Confusion matrix oluştur
figure;
confusionchart(test_labels,predictions);
title('Confusion Matrix for Test Data');







