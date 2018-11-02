import os, sys, face_recognition, pickle
from flask import Flask, request, render_template, send_from_directory
from sightengine.client import SightengineClient

client = SightengineClient('972611969', 'WfAdW4emyXWCYpioJ8yt')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.basename('uploads')

valid_imgs_exts = ['.jpg', '.jpeg', '.png', '.bmp']
distance_threshold = 0.6
knn_clf = pickle.load(open('thot_model.clf', 'rb'))

@app.route('/')
def root():
  return render_template('index.html')

@app.route('/uploads/<filename>')
def send_image(filename):
  return send_from_directory('uploads', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
  file = request.files['image']
  filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
  if not os.path.splitext(filename)[-1] in valid_imgs_exts:
    return render_template('index.html', init=True, invalid=True)
  file.save(filename)
  
  output = client.check('face-attributes').set_file(filename)
  if output['status'] == 'success':
    if 'faces' in output:
      for face in output['faces']:
        if 'attributes' in face and face['attributes']['minor'] >= 0.85:
            os.remove(filename)
            return render_template('index.html', init=True, pedo=True)
  
  X_img = face_recognition.load_image_file(filename)
  X_face_locations = face_recognition.face_locations(X_img)
  
  if len(X_face_locations) == 0:
    return render_template('index.html', init=True, invalid=True)
    
  faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
  closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
  are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

  return render_template('index.html', init=True, path=file.filename, thot=', '.join([x[0] for x in [(pred, loc) if rec else ('Unknown', loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)] if not x[0] == 'Unknown']))

if __name__ == '__main__':
  app.run(debug=True)
