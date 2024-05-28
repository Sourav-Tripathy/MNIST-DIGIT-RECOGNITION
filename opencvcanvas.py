import cv2
import numpy as np
from keras.models import load_model

# Load the trained LeNet-5 model
model = load_model('lenet5_model.h5')n

drawing = False
last_point = None

def draw_on_canvas(event, x, y, flags, param):
    global drawing, last_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, last_point, (x, y), (255, 255, 255), 20)
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        
        drawing = False
        cv2.line(canvas, last_point, (x, y), (255, 255, 255), 20)

canvas = np.zeros((280, 280, 3), dtype=np.uint8)
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', draw_on_canvas)

#prediction button and a clear button
cv2.putText(canvas, 'Predict (Press "p")', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(canvas, 'Clear (Press "c")', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

while True:
    cv2.imshow('Canvas', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas[:] = 0
        cv2.putText(canvas, 'Predict (Press "p")', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'Clear (Press "c")', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif key == ord('p'):
        # Preprocess the drawn digit
        digit = cv2.resize(canvas, (28, 28))
        digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        digit = digit / 255.0
        digit = np.expand_dims(digit, axis=0)
        digit = np.expand_dims(digit, axis=-1)
        
        # Predict the digit using the LeNet-5 model
        prediction = model.predict(digit)
        predicted_digit = np.argmax(prediction)
        
        # Display the predicted digit
        cv2.putText(canvas, f'Predicted digit: {predicted_digit}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()