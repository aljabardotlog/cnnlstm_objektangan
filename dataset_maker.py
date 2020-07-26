import cv2
import imutils

bg = None

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    image_num = 0

    start_recording = False
    while(True):
        grabbed, frame = camera.read()
        if (grabbed == True):
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            if num_frames < 5:
                run_avg(gray, aWeight)
            else:
                hand = segment(gray)
                if hand is not None:
                    (thresholded, segmented) = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    if start_recording:
                        cv2.imwrite("dataset/validation/kertas/kertas" + str(image_num) + '.png', thresholded)
                        image_num += 1
                    cv2.imshow("Thesholded", thresholded)

            cv2.rectangle(clone, (left, top), (right, bottom), (255,255,0), 1)

            num_frames += 1

            cv2.imshow("Video Feed", clone)
            keypress = cv2.waitKey(1)

            if keypress == ord("q") or image_num > 50:
                print("Stop Recording")
                break
        
            if keypress == ord("s"):
                print("Start Recording")
                start_recording = True

        else:
            print("[Warning!] Error input, Please check your camera")
            break
main()