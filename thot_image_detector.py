import face_recognition, pickle, cv2

print("Reading data from thot_data.pickle...")
data = pickle.loads(open("thot_data.pickle", "rb").read())
img = cv2.imread("test3.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb, model="cnn")
encodings = face_recognition.face_encodings(rgb, locations)
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
    print("It's {}".format(name))
