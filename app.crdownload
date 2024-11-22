from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask application
app = Flask(__name__)

# Simulate some sample data (Temperature and Bird Population)
data = {
    'Temperature': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Rainfall': [80, 85, 90, 95, 100, 105, 110, 115, 120, 125],
    'Bird_Population': [1000, 950, 920, 900, 880, 860, 850, 830, 810, 800]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['Temperature', 'Rainfall']]
y = df['Bird_Population']

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X, y)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict the bird population based on temperature
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input temperature from the form
        temp = float(request.form['temperature'])
        
        # Predict the bird population based on temperature
        predicted_population = model.predict([[temp, 100]])  # Assuming average rainfall is 100 for simplicity
        
        # Create a plot to visualize the relationship between temperature and bird population
        plt.figure(figsize=(6,4))
        plt.scatter(df['Temperature'], df['Bird_Population'], color='blue', label='True Values')
        plt.plot(df['Temperature'], model.predict(X), color='red', label='Fitted Line')
        plt.xlabel('Temperature')
        plt.ylabel('Bird Population')
        plt.title('Impact of Temperature on Bird Population')
        plt.legend()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode the image to display in HTML
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        return render_template('result.html', predicted_population=predicted_population[0], img_base64=img_base64)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
