import random

# Generate synthetic dataset
def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        age = random.randint(20, 80)
        bmi = round(random.uniform(18.5, 40.0), 1)
        blood_pressure = random.randint(80, 180)
        glucose = random.randint(70, 200)
        # Simple rule: higher BMI and glucose increase disease risk
        risk = 1 if (bmi > 30 and glucose > 130) else 0
        data.append((age, bmi, blood_pressure, glucose, risk))
    return data

# Split data into training and testing sets
def split_data(data, train_ratio=0.8):
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

# Train a simple model (calculate average values for each class)
def train_model(training_data):
    sums = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    counts = {0: 0, 1: 0}
    for age, bmi, bp, glucose, risk in training_data:
        sums[risk][0] += age
        sums[risk][1] += bmi
        sums[risk][2] += bp
        sums[risk][3] += glucose
        counts[risk] += 1
    averages = {
        risk: [s / counts[risk] for s in sums[risk]]
        for risk in [0, 1]
    }
    return averages

# Predict risk based on which average the input is closer to
def predict(model, sample):
    distances = {}
    for risk in model:
        distance = sum(
            abs(a - b) for a, b in zip(sample, model[risk])
        )
        distances[risk] = distance
    return min(distances, key=distances.get)

# Evaluate model accuracy
def evaluate_model(model, testing_data):
    correct = 0
    for age, bmi, bp, glucose, risk in testing_data:
        prediction = predict(model, (age, bmi, bp, glucose))
        if prediction == risk:
            correct += 1
    accuracy = correct / len(testing_data)
    return accuracy

# Main function
def main():
    data = generate_data(100)
    training_data, testing_data = split_data(data)
    model = train_model(training_data)
    accuracy = evaluate_model(model, testing_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Example prediction
    new_patient = (45, 28.0, 120, 140)  # age, bmi, bp, glucose
    prediction = predict(model, new_patient)
    print("Prediction for new patient:", "Disease" if prediction == 1 else "No Disease")

if __name__ == "__main__":
    main()
