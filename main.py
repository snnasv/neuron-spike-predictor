
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def did_neuron_spike(threshold, simulation_time, input_current, leak_strength):
    voltage = 0
    resting_voltage = 0


    for time in range(simulation_time):


        leak_current = leak_strength * (voltage - resting_voltage)
        voltage = voltage + input_current - leak_current


        if voltage >= threshold:
            return 1
        

    return 0








def generate_dataset():
    X = []
    y = []

    for i in range(1, 31):
        input_current = i / 10

        for j in range(1, 21):
            leak_strength = j / 100
            spike = did_neuron_spike(10,30,input_current, leak_strength)
            X.append([input_current, leak_strength])
            y.append(spike)

    return X, y







X, y = generate_dataset()
print("Number of input examples: ", len(X))
print("Number of labels: ", len(y))



print("First 5 examples:")

k = 0

while k < 5:
    print(X[k])
    print(y[k])
    k += 1


summ = 0
for e in y:
    summ += e

print("Number of Spikes: ", summ)





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)




def print_prediction(model, input_current, leak_strength):
    prediction = model.predict([[input_current, leak_strength]])


    if prediction[0] == 1:
        print("Input current: ", input_current, "Leak strength: ", leak_strength, "Spike predicted!")
    else:
        print("Input current: ", input_current, "Leak strength: ", leak_strength, "No spike predicted!")

    
print_prediction(model, 1.5, 0.1)
print_prediction(model, 0.5, 0.1)
print_prediction(model, 2.5, 0.2)

