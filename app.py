import os
import secrets
from flask import Flask, request, render_template, session, redirect
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from src.model import EstimateModel
from wtforms import SubmitField, IntegerField
from wtforms.fields import DateField
from wtforms.validators import DataRequired, NumberRange

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'geojson'}
MODEL_PATH = 'data/model/stacking.pkl'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class InfoForm(FlaskForm):
    startdate = DateField('Start Date', format='%Y-%m-%d', validators=(DataRequired(),))
    enddate = DateField('End Date', format='%Y-%m-%d', validators=(DataRequired(),))
    cloud_coverage = IntegerField('Cloud Coverage', validators=[NumberRange(min=0, max=100)])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error_message = None
    
    form = InfoForm()

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
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER']), exist_ok=True)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                if form.validate_on_submit():
                    session['startdate'] = form.startdate.data
                    session['enddate'] = form.enddate.data
                    return redirect('date')

                # Use the uploaded file and selected dates for prediction
                model = EstimateModel(MODEL_PATH)
                results = model.predict(file_path, session.get('startdate'), session.get('enddate'))

                # Pass the results to the results.html template
                return render_template('results.html', results=results)

    return render_template('upload.html', error_message=error_message, form=form)

if __name__ == '__main__':
    app.run(debug=True)
