from flask import Flask, request, jsonify

app = Flask(__name__)

# Endpoint to receive POST requests with audio file
@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if file is present and has an allowed extension
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Assuming the script processes the audio file
    # Replace this with your actual processing logic
    result = process_audio_file(file)
    
    return jsonify({'result': result}), 200

def process_audio_file(file):
    # Replace this function with your actual audio processing logic
    # For example, you can use libraries like librosa or PyDub
    # Here we'll just return a placeholder result
    return "Audio file processed successfully"

if __name__ == '__main__':
    app.run(debug=True)
