from flask import Flask, render_template, request
from model import EstimateModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def estimation():
    if request.method == 'POST':
        file = request.files['file']
        
        # TODO: write correct path
        temp_path = 'data/file.shp'
        file.save(temp_path)
        print("File saved successfully")
        
        # Use the EstimateModel class
        print("Load model")
        model = EstimateModel()
        print("Wait for model predictions")
        results = model.predict(temp_path)
        
        # Return the file for download along with the results
        return render_template('results.html', results=results)
    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
