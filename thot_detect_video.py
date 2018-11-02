import os, sys, time, math, cv2, imutils, face_recognition, pickle, queue, threading, numpy

DIST_THRESHOLD = .6
NUM_THREADS = 8

knn_clf = pickle.load(open('thot_model.clf', 'rb'))
video = cv2.VideoCapture(sys.argv[0])
if not video:
  print("Failed to open video...")
  sys.exit(-1)
fps = math.ceil(video.get(cv2.CAP_PROP_FPS))

q = queue.Queue(maxsize=NUM_THREADS)
q_lock = threading.Lock()

thots_found = {}
valid_faces = 0

def queue_job(worker):
  X_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  X_img = imutils.resize(frame, width=640)
  X_face_locations = face_recognition.face_locations(X_img)
  if not X_face_locations:
    return

  faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
  
  closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
  are_matches = [closest_distances[0][i][0] <= DIST_THRESHOLD for i in range(len(X_face_locations))]

  with q_lock:
    for x in [(pred, loc) if rec else (None, loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]:
      y = x[0]
      if y:
        if not y in thots_found:
          thots_found[y] = 0
        thots_found[y] += 1


def queue_worker():
  while True:
    queue_job(q.get())
    q.task_done()

for _ in range(NUM_THREADS):
  t = threading.Thread(target=queue_worker)
  t.daemon = True
  t.start()

skip_frames = 0
while True:
  ret, frame = video.read()
  if not ret:
    break
  
  if not skip_frames:
    skip_frames = fps
    q.put(frame)
  else:
    skip_frames -= 1

q.join()

quarter_total_faces = (sum(thots_found.values()) / 2) / 2
for thot, total in thots_found.items():
  if total > quarter_total_faces:
    print(thot)
