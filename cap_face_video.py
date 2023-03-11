import cv2
import os
import datetime

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

def createDirectory():
    if not os.path.exists('Daten'):
        os.makedirs('Daten')
    if not os.path.exists('Daten/Aufnahmen/Gesicht/Video'):
        os.makedirs('Daten/Aufnahmen/Gesicht/Video')

def capture_video():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    out = cv2.VideoWriter(f"Daten/Aufnahmen/Gesicht/Video/{date_string}.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

    # Liste der Haar-Cascade-Dateipfade
    cascade_files = [cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                     cv2.data.haarcascades + 'haarcascade_profileface.xml']

    start_recording = False

    while True:
        ret, frame = cap.read()

        if ret:

            # Gesichtserkennung mit allen Haar-Cascade-Dateien durchführen
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            alpha = 1 # Kontrast
            beta = 8 # Helligkeit
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            flipped = cv2.flip(gray, 1)  # Bild horizontal spiegeln
            for cascade_file in cascade_files:
                face_cascade = cv2.CascadeClassifier(cascade_file)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                flipped_faces = face_cascade.detectMultiScale(flipped, 1.3, 5)  # Gesichtserkennung im gespiegelten Bild durchführen

            
                for (x, y, w, h) in faces:
                    # Zoom auf das Gesicht
                    zoom = 2
                    x_center = x + w/2
                    y_center = y + h/2
                    x_zoom = max(0, int(x_center - (w*zoom)/2))
                    y_zoom = max(0, int(y_center - (h*zoom)/2))
                    w_zoom = min(int(w*zoom), frame.shape[1]-x_zoom)
                    h_zoom = min(int(h*zoom), frame.shape[0]-y_zoom)
                    frame_zoomed = frame[y_zoom:y_zoom+h_zoom, x_zoom:x_zoom+w_zoom]
                    frame_zoomed = cv2.resize(frame_zoomed, (frame.shape[1], frame.shape[0]))
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', (frame_zoomed.shape[1], frame_zoomed.shape[0]))
                    cv2.imshow('frame', frame_zoomed)

                else:
                    #cv2.putText(frame, "Bitte drehen Sie Ihren Kopf um 90 Grad nach links und drücken Sie erneut Enter, um die Aufnahme zu beenden", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if start_recording:
                        out.write(frame_zoomed)

                for (x, y, w, h) in flipped_faces:
                    # Die Koordinaten des Rechtecks anpassen, da das Bild horizontal gespiegelt wurde
                    x = frame.shape[1] - x - w
                    # Zoom auf das Gesicht
                    zoom = 2
                    x_center = x + w/2
                    y_center = y + h/2
                    x_zoom = max(0, int(x_center - (w*zoom)/2))
                    y_zoom = max(0, int(y_center - (h*zoom)/2))
                    w_zoom = min(int(w*zoom), frame.shape[1]-x_zoom)
                    h_zoom = min(int(h*zoom), frame.shape[0]-y_zoom)
                    frame_zoomed = frame[y_zoom:y_zoom+h_zoom, x_zoom:x_zoom+w_zoom]
                    frame_zoomed = cv2.resize(frame_zoomed, (frame.shape[1], frame.shape[0]))
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', (frame_zoomed.shape[1], frame_zoomed.shape[0]))
                    cv2.imshow('frame', frame_zoomed)
                    if start_recording:
                        out.write(frame_zoomed)
                        
            if cv2.waitKey(1) == 13:
                if not start_recording:
                    start_recording = True
                else:
                    break

        else:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


createDirectory()
capture_video()
