#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>
#include "imgui.h"

using namespace std;
using namespace cv;

bool captureImage = false;
Mat capturedImage;
string enteredName;

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        captureImage = true;
    }
}

void detectEyes(Mat &img, Rect face, CascadeClassifier &eyes_cascade)
{
    Mat ROI = img(face);
    vector<Rect> eyes;
    eyes_cascade.detectMultiScale(ROI, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const Rect &e : eyes)
    {
        Rect eyeRect(face.x + e.x, face.y + e.y, e.width, e.height);
        rectangle(img, eyeRect.tl(), eyeRect.br(), Scalar(0, 255, 0), 2);
    }
}

void promptForName(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        // Display a prompt window for entering the name
        enteredName = cv::getWindowProperty("EnterName", WND_PROP_AUTOSIZE) == WINDOW_NORMAL ? "" : enteredName;
        enteredName += cv::waitKey(0);

        putText(capturedImage, enteredName, Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
        imshow("Camera", capturedImage);
    }
}

int main()
{
    namedWindow("Camera");
    VideoCapture video(0);
    CascadeClassifier facedetect;
    CascadeClassifier eyes_cascade;

    if (!video.isOpened())
    {
        cout << "Error: Couldn't open the camera." << endl;
        return -1;
    }

    setMouseCallback("Camera", onMouse, NULL);

    if (!facedetect.load("C:/Users/Bibek Joshi/Desktop/ImageRecognition/data/haarcascades/lbpcascade_frontalface_improved.xml") ||
        !eyes_cascade.load("C:/Users/Bibek Joshi/Desktop/ImageRecognition/data/haarcascades/haarcascade_eye.xml"))
    {
        cout << "Error: Couldn't load one or more cascade classifiers." << endl;
        return -1;
    }

    Mat img;
    ofstream csvFile("C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/data.csv", ios::app);

    while (true)
    {
        video.read(img);

        // Check if the frame is empty
        if (img.empty())
        {
            cout << "Error: Couldn't read frame from the camera." << endl;
            break;
        }

        if (captureImage)
        {
            // Copy the image for display in the prompt window
            capturedImage = img.clone();

            // Display a window to prompt the user to enter a name
            namedWindow("EnterName");
            setMouseCallback("EnterName", promptForName, NULL);

            // Wait for the user to enter a name
            imshow("EnterName", capturedImage);
            waitKey(0);

            // Save image
            string imageName = "C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/" + enteredName + "_captured_photo.jpg";
            imwrite(imageName, capturedImage);
            cout << "Photo captured and saved as " << imageName << endl;

            // Save information to CSV file
            ofstream csvFile("C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/data.csv", ios::app);
            csvFile << enteredName << "," << imageName << endl;
            csvFile.close();

            // Close the prompt window
            destroyWindow("EnterName");

            captureImage = false;
        }

        vector<Rect> faces;
        facedetect.detectMultiScale(img, faces, 1.3, 5);

        cout << faces.size() << " face(s) found." << endl;

        for (const Rect &face : faces)
        {
            // Use the rectangle function from the imgproc namespace
            rectangle(img, face.tl(), face.br(), Scalar(50, 50, 255), 2);
            cv::rectangle(img, Point(0, 0), Point(250, 70), Scalar(50, 50, 255), FILLED);
            putText(img, to_string(faces.size()) + " Face Found", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
            // Draw facial features
            detectEyes(img, face, eyes_cascade);
        }

        imshow("Camera", img);

        char c = waitKey(25);
        if (c == 27) // Press 'Esc' key to exit
            break;
    }

    csvFile.close();
    return 0;
}
