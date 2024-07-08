clear all;
clc;

% Load the dataset
filePath = 'C:\Users\ikoba\Desktop\PWr_Courses\Artificial Neural Networks\Project/sensor-fault-detection.csv';
data = readtable(filePath);

% Extract values and labels
values = data.Value;
labels = categorical(repmat("N", size(values)));

% Combine all faults into a single category 'AN' for Anomaly
% Drift Fault Detection
diffValues = diff(values);
driftThreshold = mean(diffValues) + 3 * std(diffValues);
isDrift = [false; abs(diffValues) > driftThreshold];
labels(isDrift) = 'AN';

% Bias Fault Detection
baselineValue = 25;
biasThreshold = 5;
isBias = abs(values - baselineValue) > biasThreshold;
labels(isBias) = 'AN';

% Precision Degradation Fault Detection
windowSize = 50;
rollingVar = movvar(values, windowSize);
precisionThreshold = 2 * var(values);
isPrecisionDegradation = rollingVar > precisionThreshold;
labels(isPrecisionDegradation) = 'AN';

% Spike Fault Detection
spikeThreshold = mean(values) + 5 * std(values);
isSpike = abs(values - mean(values)) > spikeThreshold;
labels(isSpike) = 'AN';

% Stuck Fault Detection
stuckDuration = 100;
isStuck = false(size(values));
for i = 1:length(values) - stuckDuration
    if all(values(i:i+stuckDuration-1) == values(i))
        isStuck(i:i+stuckDuration-1) = true;
    end
end
labels(isStuck) = 'AN';

% Create a table with values and labels
labeledData = table(values, labels);

% Save the labeled data to a file
writetable(labeledData, 'labeled_sensor_fault_detection.csv');

% Prepare the data for deep learning
windowSize = 100;
features = [];
newLabels = [];
for i = 1:length(values) - windowSize + 1
    features = [features; values(i:i+windowSize-1)'];
    newLabels = [newLabels; mode(labels(i:i+windowSize-1))];
end

% Convert labels to categorical
newLabels = categorical(newLabels);

% Save the processed features and labels to a file
processedData = table(features, newLabels);
writetable(processedData, 'processed_sensor_fault_detection.csv');

% Shuffle the data
numSamples = size(features, 1);
shuffledIndices = randperm(numSamples);
shuffledFeatures = features(shuffledIndices, :);
shuffledLabels = newLabels(shuffledIndices);

% Split the data into training (70%), validation (10%), and test (20%) sets
trainRatio = 0.7;
valRatio = 0.1;
testRatio = 0.2;

trainEnd = floor(trainRatio * numSamples);
valEnd = trainEnd + floor(valRatio * numSamples);

trainFeatures = shuffledFeatures(1:trainEnd, :);
trainLabels = shuffledLabels(1:trainEnd);

valFeatures = shuffledFeatures(trainEnd+1:valEnd, :);
valLabels = shuffledLabels(trainEnd+1:valEnd);

testFeatures = shuffledFeatures(valEnd+1:end, :);
testLabels = shuffledLabels(valEnd+1:end);

% Reshape the features for CNN
trainFeatures = reshape(trainFeatures', [1, windowSize, 1, size(trainFeatures, 1)]);
valFeatures = reshape(valFeatures', [1, windowSize, 1, size(valFeatures, 1)]);
testFeatures = reshape(testFeatures', [1, windowSize, 1, size(testFeatures, 1)]);

% Define the CNN layers with dropout layers to reduce overfitting
layers = [
    imageInputLayer([1 windowSize 1], 'Name', 'input')
    convolution2dLayer([1 3], 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'maxpool1')
    dropoutLayer(0.2, 'Name', 'dropout1')
    convolution2dLayer([1 3], 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'maxpool2')
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(2, 'Name', 'fc') % Number of classes (N and AN)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% Define the training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valFeatures, valLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the CNN
[net, info] = trainNetwork(trainFeatures, trainLabels, layers, options);

% Evaluate the trained model on the validation set
predictedValLabels = classify(net, valFeatures);
valAccuracy = sum(predictedValLabels == valLabels) / numel(valLabels);
disp(['Validation Accuracy: ', num2str(valAccuracy * 100), '%']);

% Evaluate the trained model on the test set
predictedTestLabels = classify(net, testFeatures);
testAccuracy = sum(predictedTestLabels == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(testAccuracy * 100), '%']);

% Plot training and validation accuracy and loss
figure;
subplot(2,1,1);
plot(info.TrainingAccuracy, 'LineWidth', 2);
hold on;
plot(info.ValidationAccuracy, 'LineWidth', 2);
title('Accuracy');
xlabel('Iteration');
ylabel('Accuracy');
legend('Training Accuracy', 'Validation Accuracy');
grid on;

subplot(2,1,2);
plot(info.TrainingLoss, 'LineWidth', 2);
hold on;
plot(info.ValidationLoss, 'LineWidth', 2);
title('Loss');
xlabel('Iteration');
ylabel('Loss');
legend('Training Loss', 'Validation Loss');
grid on;

% Save the figure
saveas(gcf, 'training_validation_performance.png');

% Confusion matrix for the test set
figure;
cm = confusionchart(testLabels, predictedTestLabels);
cm.Title = 'Confusion Matrix for Test Set';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf, 'confusion_matrix.png');
