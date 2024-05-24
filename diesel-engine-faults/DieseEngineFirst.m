clc;
clear;

data = load('Features_2500RPM_0dB_full.mat'); 
DataBase_table = data.DataBase_table; 

%disp(size(DataBase_table));
%summary(DataBase_table);
features = DataBase_table(:,1:84); % Assuming the first 84 columns are features
featuresMatrix = table2array(features); % Convert table to array for processing

labels = [ones(250, 1); 2*ones(250, 1); 3*ones(1500, 1); 4*ones(1500, 1)];  %label
labels = categorical(labels, 1:4, {'NormalOperation', 'PressureReduction', 'CompressionReduction', 'FuelReduction'});

shuffled_indices = randperm(height(features));
shuffled_features = featuresMatrix(shuffled_indices, :);
shuffled_labels = labels(shuffled_indices);

% Ratio of train, validation and test data
%total_samples = height(shuffled_data);
%train_ratio = 0.7;
%validation_ratio = 0.1;
%test_ratio = 0.2;

% seperate data according to train,validation and test subsets
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

% Deep CNN model 
layers = [
    imageInputLayer([1 84 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    convolution2dLayer(3, 16, 'Padding', 'same')
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