from flask import Flask, request, render_template, url_for, send_from_directory
import os
from predict import predict_leaf_health

def remove_all_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate through all files
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        try:
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                print(f"Skipped directory: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    remove_all_files("uploads")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = predict_leaf_health(filepath)
        image_url = url_for('uploaded_file', filename=file.filename)
        return render_template('result.html', result=result, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
