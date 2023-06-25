import os
import secrets
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from src.model import EstimateModel

UPLOAD_FOLDER = 'data/nature_reserves'
ALLOWED_EXTENSIONS = {'geojson'}

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error_message = None

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            error_message = 'No file part'
        else:
            file = request.files['file']
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                error_message = 'No selected file'
            elif not allowed_file(file.filename):
                error_message = 'File extension not allowed'
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Use the uploaded file for prediction
                model = EstimateModel('data/model')
                results = model.predict(file_path)

                # Pass the results to the results.html template
                return render_template('results.html', results=results)

    return render_template('upload.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
