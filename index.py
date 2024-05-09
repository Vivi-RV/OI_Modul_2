import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Завантаження даних
train_data = scipy.io.loadmat('train_digits.mat')
test_data = scipy.io.loadmat('test_digits.mat')

X_train = train_data['X']
y_train = train_data['y'].ravel()
X_test = test_data['X']
y_test = test_data['y'].ravel()

# Нормалізація даних
X_train /= 255.0
X_test /= 255.0

# Розділення на навчальну та тестову вибірки
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Моделі для 2-шарової нейронної мережі з різними значеннями активації
models_2_layer = [
    MLPClassifier(hidden_layer_sizes=(3, 3), solver='lbfgs', activation='logistic', max_iter=200, random_state=42),
    MLPClassifier(hidden_layer_sizes=(3, 3), solver='lbfgs', activation='tanh', max_iter=200, random_state=42)
]


# Моделі для 3-шарової нейронної мережі з різними значеннями активації
models_3_layer = [
    MLPClassifier(hidden_layer_sizes=(20, 7, 10), solver='lbfgs', activation='logistic', max_iter=200, random_state=42),
    MLPClassifier(hidden_layer_sizes=(20, 7, 10), solver='lbfgs', activation='tanh', max_iter=200, random_state=42),
    MLPClassifier(hidden_layer_sizes=(20, 7, 10), solver='lbfgs', activation='relu', max_iter=200, random_state=42)
]

# Навчання та оцінка моделей для 2-шарової нейронної мережі
for model in models_2_layer:
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy for 2-layer model with activation", model.activation, ":", accuracy)

# Навчання та оцінка моделей для 3-шарової нейронної мережі
for model in models_3_layer:
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy for 3-layer model with activation", model.activation, ":", accuracy)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

# Візуалізація результатів
labels = ['SVM', '2-Layer (logistic, logistic)', '2-Layer (tanh, tanh)', '3-Layer (logistic, tanh, logistic)', '3-Layer (logistic, tanh, tanh)', '3-Layer (logistic, tanh, relu)']
accuracies = [svm_accuracy] + [accuracy_score(y_test, model.predict(X_test)) for model in models_2_layer + models_3_layer]
plt.bar(labels, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xticks(rotation=45, ha='right')
plt.show()