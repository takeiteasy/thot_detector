import os, sys, time, cv2, imutils, face_recognition, pickle

distance_threshold = 0.6
knn_clf = pickle.load(open('thot_model.clf', 'rb'))

video = cv2.VideoCapture("test.mp4")
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
while True:
    ret, frame = video.read()
    X_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    X_img = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(X_img.shape[1])
    frame_number += 1

    X_face_locations = face_recognition.face_locations(X_img)

    if not X_face_locations:
        continue
    
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    print([(pred, loc) if rec else ('Unknown', loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)])
