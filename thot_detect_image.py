import os, sys, time, face_recognition, pickle

distance_threshold = 0.6
knn_clf = pickle.load(open('thot_model.clf', 'rb'))

for fn in sys.stdin:
    start = time.time()

    fn = fn.rstrip()
    print('Testing: ' + fn)

    X_img = face_recognition.load_image_file(fn)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        print([])

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    print([(pred, loc) if rec else ('Unknown', loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)])

    end = time.time()
    print('Taking ' + str(end - start) + ' seconds')

