from requests_html import HTMLSession
from sklearn import neighbors
import sys, os, re, time, math, pickle, cv2, face_recognition

img_url_re = re.compile(r'https:\/\/ci\.phncdn\.com\/pics\/pornstars\/\d{3}\/\d{3}\/\d{3}\/\(\S+\)thumb_\d+\.jpg')

def generate_url(name):
    return "https://www.pornhub.com/pornstar/{}/official_photos".format(name.lower().replace(' ', '-'))

def get_image_urls(session, url, try_again=True):
    r = session.get(url)
    if not r.status_code == 200:
        return get_image_urls(session, url)
    r.html.render()
    imgs = []
    tags = r.html.find('img')
    for x in ['src', 'data-thumb_url']:
        for tag in tags:
            if x in tag.attrs and img_url_re.match(tag.attrs[x]):
                imgs.append(tag.attrs[x])

    imgs = list(set(imgs))
    if not imgs:
        if try_again:
            return get_image_urls(session, url, False)
        else:
            return None
    return imgs

def download_images(session, thot, urls):
    ret = []
    for url in urls:
        r = session.get(url, stream=True)
        if r.status_code == 200:
            try:
                path = "thots/" + name + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                path += url.split('_')[-1]

                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)

                if len(face_recognition.face_locations(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), model="cnn")) == 1:
                    ret.append(path)
                else:
                    os.remove(path)
            except:
                pass
    return ret

session = HTMLSession()

if not os.path.exists("thots"):
    os.makedirs("thots")

with open('thots.txt') as fp:
    content = fp.readlines()

thot_map = {}
for name in content:
    name = name.rstrip()
    if os.path.exists("thots/" + name):
        continue
    print("Downloading images for: {}".format(name))
    url = generate_url(name)
    urls = get_image_urls(session, url)
    if not urls:
        print("Failed to get results for: {}".format(name))
    else:
        imgs = download_images(session, name, urls)
        if not imgs:
            print("Failed to get results for: {}".format(name))
            os.rmdir(name)
        else:
            print("Downloaded {} of {} images for: {}".format(len(imgs), len(urls), name))
            thot_map[name] = imgs
    time.sleep(1)
session.close()

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
