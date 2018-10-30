from sklearn import neighbors
import face_recognition, pickle, cv2, os, sys, math

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

X = []
y = []

for dir in os.listdir('thots'):
    path = os.path.join('thots', dir)
    if not os.path.isdir(path):
        continue
    for img in os.listdir(path):
        if not os.path.splitext(img)[-1] in ALLOWED_EXTENSIONS:
            continue

        img_path = os.path.join(path, img)
        image = face_recognition.load_image_file(img_path)
        boxes = face_recognition.face_locations(image)
        if len(boxes) == 1:
            X.append(face_recognition.face_encodings(image, known_face_locations=boxes)[0])
            y.append(dir)

n_neighbors = int(round(math.sqrt(len(X))))
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
knn_clf.fit(X, y)

with open('thot_model.clf', 'wb') as f:
    pickle.dump(knn_clf, f)
