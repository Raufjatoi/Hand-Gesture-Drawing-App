# Hand Gesture Drawing App

This project is a Hand Gesture Drawing App using OpenCV, Mediapipe, and Numpy. It allows you to draw on the screen using hand gestures detected via your webcam.

## Features

- Draw with different colors (Blue, Green, Red, Yellow, Orange, Purple, Pink, Sky Blue)
- Adjust drawing pen thickness (2px, 5px, 10px)
- Draw shapes such as circles and rectangles
- Clear the canvas
- Toggle between drawing and erasing modes

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Raufjatoi/object-drawing-app.git
    cd object-detection-app
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your webcam is connected and functioning properly.
2. Run the following command to start the application:
    ```bash
    python app4.py or app2.py ( i suggest ya to try app2 cause app4 is new variotion of it and its still underwork)
    ```
3. A new window will open showing the webcam feed with the drawing canvas overlay.
4. Use your hand gestures to draw on the canvas:
   - **Hand Gestures**: Move your index finger to draw lines. Close your index finger and thumb to stop drawing.
   - **Colors**: Select different colors by touching the corresponding color boxes on the left side of the screen.
   - **Pen Thickness**: Press `1` for a thin pen (2 pixels), `2` for a medium pen (5 pixels), and `3` for a thick pen (10 pixels).
   - **Shapes**: Draw circles and rectangles by selecting the respective shape buttons on the top right of the screen.
   - **Clear Canvas**: Clear the canvas by touching the "CLEAR" button at the top left of the screen.

5. Press `q` to quit the application.

## Example

![Hand Gesture Drawing App](drawing_app_example.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
