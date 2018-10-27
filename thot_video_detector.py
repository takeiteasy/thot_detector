import face_recognition, pickle, cv2, imutils

data = pickle.loads(open("thot_data.pickle", "rb").read())
video = cv2.VideoCapture("test.mp4")
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    frame_number += 1

    locations = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, locations)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            ids = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in ids:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    if names:
        print("{}: {}".format(frame_number, names))

