

from flask import Flask, render_template, Response
from deepface import DeepFace as dep
import cv2
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
app=Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            result=dep.analyze(frame,actions=['emotion','gender'],enforce_detection=False)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.1,4)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,result['dominant_emotion'],(0,50),font,1,(0,0,255),1,cv2.LINE_4);
            cv2.putText(frame,result['gender'],(1,90),font,1,(0,0,255),1,cv2.LINE_4);
            

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)