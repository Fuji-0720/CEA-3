import os
import cv2
import pickle

def main():
    cap = cv2.VideoCapture(1)

    # Get frame size (width and height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = (width, height)

    os.makedirs('./calib_files', exist_ok=True)
    os.makedirs('./images', exist_ok=True)

    # Save frame size to a file
    with open('./calib_files/frameSize.pkl', 'wb') as f:
        pickle.dump(frameSize, f)

    num = 0
    while cap.isOpened():
        success, img = cap.read()
        k = cv2.waitKey(5)
        if k == 27 or  k == ord('q'):
            break
        elif k == ord('s'): 
            cv2.imwrite('./images/img' + str(num) + '.png', img)
            num += 1
        cv2.imshow("Press S to Capture images", img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()