import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from src.model import EstimateModel

UPLOAD_FOLDER = 'data/nature_reserves'
ALLOWED_EXTENSIONS = {'geojson'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, display a message
        if file.filename == '':
            flash('Please select and upload a file')
            return redirect(request.url)
        # If the file extension is not allowed, display a message
        if not allowed_file(file.filename):
            flash('File extension not allowed')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Use the uploaded file for prediction
            model = EstimateModel('data/model/stacking.pkl')
            results = model.predict(file_path)
            
            # Pass the results to the results.html template
            return render_template('results.html', results=results)
    return render_template('upload.html')        
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''

if __name__ == '__main__':
    app.run(debug=True)
