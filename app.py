import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime, timedelta
import numpy as np
import face_recognition
import base64
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from connect import conn

#### Defining Flask App
app = Flask(__name__)
directory2 = os.path.join(app.static_folder,
                          'face-recog-attendance-sy-1d97a-firebase-adminsdk-l3ltw-7df251c3f9.json')
# Initialize Firebase credentials
cred = credentials.Certificate(directory2)
firebase_admin.initialize_app(cred, {'storageBucket': 'face-recog-attendance-sy-1d97a.appspot.com'})
namem = ""
roll = ""
class_names = []
encode_known = []

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

date_today = datetoday()


#### get a number of total registered users
def totalreg():
    bucket = storage.bucket()
    blob_list = bucket.list_blobs(prefix='faces/')
    count = sum(1 for blob in blob_list if blob.name.endswith('.jpg'))
    return count


#### Identify face using ML model
def identify_face(facearray):
    for i, enc in enumerate(encode_known):
    if face_recognition.compare_faces([enc], facearray)[0]:
        return class_names[i]
    return None

@app.route('/train')
#### A function which trains the model on all the faces available in faces folder
def train_model():
    global class_names, encode_known
    class_names = []
    encode_known = []
    bucket = storage.bucket()
    blob_list = bucket.list_blobs(prefix='faces/')
    # Loop through blobs and encode faces
    for blob in blob_list:
        # Download image file to memory and read with OpenCV
        img_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        class_names.append(blob.name[6:-4])

        # Convert color format and locate face in the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img)

        # Encode face and add to encodings list
        encodes_cur_frame = face_recognition.face_encodings(img, boxes)[0]
        encode_known.append(encodes_cur_frame)


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    results = conn.read(f"SELECT * FROM \"{date_today}\"")
    return results


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = (datetime.utcnow()+timedelta(hours=5.5)).strftime("%H:%M:%S")

    exists = conn.read(f'SELECT EXISTS(SELECT * FROM \"{date_today}\" WHERE roll={userid})')
    if exists[0][0] == 0:
        try:
            conn.insert(f'INSERT INTO \"{date_today}\"(name, roll, time) VALUES (%s, %s, %s)', (username, userid, current_time))
            print('Added Data in Database')
        except Exception as e:
            print(e)



################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    #no_of_tables = cur.execute('SHOW TABLES')
    #print('Tables', no_of_tables)
    conn.create(f'CREATE TABLE IF NOT EXISTS \"{date_today}\"(name VARCHAR(20), roll INT, time TIME)')
    userDetails = extract_attendance()
    return render_template('home.html', l=len(userDetails), totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)

@app.route('/video')
def video():
    return render_template('video.html')


# This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET', 'POST'])
def start():
    image_data = request.json['image']
    # Decode Base64-encoded image data and convert to NumPy array
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Convert image from BGR (OpenCV default) to RGB (face_recognition default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get face locations in the image
    face_locations = face_recognition.face_locations(img_rgb)

    # Get face encodings from the locations
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)[0]
    print(face_encodings)
    train_model()
    identified_person = identify_face(face_encodings)
    if identified_person != None:
        add_attendance(identified_person)
    else:
        print('person Not identified')

    userDetails = extract_attendance()
    return render_template('home.html', l=len(userDetails), totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)


#### This function will run when we add a new user
@app.route('/start_capture', methods=['GET', 'POST'])
def start_capture():
    # Start capturing logic goes here
    global namem, roll
    namem = request.form.get('newusername')
    roll = request.form.get('newuserid')
    return render_template('capture.html')

@app.route('/save', methods=['POST'])
def save():
    dataUrl = request.json['dataUrl']
    filename = f'{namem}_{roll}.jpg'
    bucket = storage.bucket()
    # Decode Base64-encoded image data and convert to NumPy array
    image_data = dataUrl
    img_bytes = base64.b64decode(image_data.split(',')[1])

    # Upload image to Firebase Storage
    blob = bucket.blob(f'faces/{filename}')
    blob.upload_from_string(img_bytes, content_type='image/jpeg')
    print(f"Image {filename} uploaded to Firebase Storage")
    return ''


#### Our main function which runs the Flask App
if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
