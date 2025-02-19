from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
#import webbrowser
import os

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        # Collect form input data
        data = CustomData(
            customer_id=request.form.get('customer_id'),
            terminal_id=request.form.get('terminal_id'),
            tx_amount=float(request.form.get('tx_amount')),
            tx_hour=int(request.form.get('tx_hour')),
            tx_day=int(request.form.get('tx_day')),
            tx_day_of_week=int(request.form.get('tx_day_of_week')),
            tx_month=int(request.form.get('tx_month')),
            is_night_tx=int(request.form.get('is_night_tx')),
            is_weekend_tx=int(request.form.get('is_weekend_tx'))
        )

        # Convert input data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:", pred_df)

        # Load prediction pipeline and predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Result:", results[0])

        # Render result in the home.html template
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    
    # host = "127.0.0.1"
    # port = 5000
    # print(f"Server is running! Open your browser and visit: http://{host}:{port}")
    
    # # Optional: Automatically open the website in the default browser
    # #webbrowser.open(f"http://{host}:{port}")

    # app.run(host=host, port=port, debug=True)

    # port = int(os.environ.get("PORT", 5000))  # Get the port dynamically from Render
    # app.run(host="0.0.0.0", port=port, debug=True)

    port = int(os.environ.get("PORT", 10000))  # Use Render’s assigned port
    app.run(host="0.0.0.0", port=port, debug=True)

    
