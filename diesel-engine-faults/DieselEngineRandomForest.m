clc;
clear;

% Veri yüklemesi
data = load('Features_2500RPM_0dB_full.mat');
DataBase_table = data.DataBase_table;
featuresMatrix = table2array(DataBase_table(:,1:84));  % Özellik matrisi

group1_features = featuresMatrix(1:250, :);
group2_features = featuresMatrix(251:500, :);
group3_features = featuresMatrix(501:2000, :);
group4_features = featuresMatrix(2001:3500, :);

% Grup tanımlamaları ve rastgele örnek seçimi
rng(1); % Rastgelelik için seed belirle
rand_indices3 = randperm(1500, 250) + 500;  % Grup 3 için rastgele satır indeksleri
rand_indices4 = randperm(1500, 250) + 2000;  % Grup 4 için rastgele satır indeksleri

selected_group3_features = group3_features(rand_indices3 - 500, :);  % Grup 3'ten rastgele seçilen satırlar
selected_group4_features = group4_features(rand_indices4 - 2000, :);  % Grup 4'ten rastgele seçilen satırlar


% Etiketlerin hazırlanması
labels = categorical([ones(250, 1); 2*ones(250, 1); 3*ones(250, 1); 4*ones(250, 1)], ...
                     1:4, {'NormalOperation', 'PressureReduction', 'CompressionReduction', 'FuelReduction'});

% Verilerin birleştirilmesi ve karıştırılması
final_features = [group1_features; group2_features; selected_group3_features; selected_group4_features];
shuffled_indices = randperm(size(final_features, 1));
shuffled_features = final_features(shuffled_indices,:);
shuffled_labels = labels(shuffled_indices);

% Eğitim ve test setlerinin ayrılması
splitIndex = floor(0.7 * size(shuffled_features, 1));
train_features = shuffled_features(1:splitIndex, :);
train_labels = shuffled_labels(1:splitIndex);
test_features = shuffled_features(splitIndex + 1:end, :);
test_labels = shuffled_labels(splitIndex + 1:end);

% Random Forest modelinin oluşturulması
numTrees = 100;
model = TreeBagger(numTrees, train_features, train_labels, 'OOBPrediction','On', ...
                  'Method','classification', 'OOBPredictorImportance','on');

% OOB Hata Oranının Çizdirilmesi
figure;
oobErrorBaggedEnsemble = oobError(model);
plot(oobErrorBaggedEnsemble)
title('Out-of-Bag Classification Error');
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');

% Test seti üzerinden tahminlerin yapılması
[predicted_labels, scores] = predict(model, test_features);
predicted_labels = categorical(predicted_labels);

% Doğruluk oranının hesaplanması
accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);


% Confusion matrix oluşturulması
figure;
confusionchart(test_labels, predicted_labels);
title('Confusion Matrix for Test Data');