import cv2
from mask_detector import detect_mask

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_mask(frame, threshold=0.4)  # Chamando com o novo threshold ajustável
        cv2.imshow('Face Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
