from requests_html import HTMLSession
import sys, os, re, time, cv2, face_recognition, pickle

img_url_re = re.compile(r'https:\/\/ci\.phncdn\.com\/pics\/pornstars\/\d{3}\/\d{3}\/\d{3}\/\(\S+\)thumb_\d+\.jpg')
session = HTMLSession()

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

def download_images(session, urls):
    ret = []
    for url in urls:
        r = session.get(url, stream=True)
        if r.status_code == 200:
            url = "/tmp/{}".format(url.split('_')[-1])
            try:
                with open(url, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                ret.append(url)
            except:
                pass
    return ret

thot_map = {}
for name in sys.stdin:
    name = name.rstrip()
    print("Downloading images for: {}".format(name))
    url = generate_url(name)
    urls = get_image_urls(session, url)
    if not urls:
        print("Failed to get results for: {}".format(name))
    else:
        imgs = download_images(session, urls)
        if not imgs:
            print("Failed to get results for: {}".format(name))
        else:
            print("Downloaded {} of {} images for: {}".format(len(imgs), len(urls), name))
            thot_map[name] = imgs
session.close()

encodings, names = [], []
for k,v in thot_map.items():
    for vv in v:
        img = cv2.imread(vv)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="cnn")
        if len(locations) == 1:
            encodings.append(face_recognition.face_encodings(rgb, locations)[0])
            names.append(k)
        os.remove(vv)

print("Encoding data to thot_data.pickle...")
data = { "encodings": encodings, "names": names }
with open("thot_data.pickle", "wb") as f:
    f.write(pickle.dumps(data))
