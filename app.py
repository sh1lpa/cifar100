from flask import Flask, request, render_template, send_from_directory

import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
application = Flask(__name__, template_folder='./src/templates/')
app = application


# route for a home page



@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' in request.files:
            image = request.files['image']
            # Save the uploaded file to a desired location
            image.save('src/uploads/' + image.filename)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(image.filename)
            return render_template('show_result.html', image_filename=image.filename, image_info=f'class of this image is predicted as {results.capitalize()}')
    # Render the upload form template for GET requests
    return render_template('upload.html')

@app.route('/src/uploads/<filename>')
def uploaded_file(filename):
    # Serve the uploaded file
    return send_from_directory('src/uploads', filename)



if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)