#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <ctime>
#include <vector>
#include <curl/curl.h>
#include <string>
// Include nlohmann/json library
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using std::string;
using std::vector;
using namespace cv;
using namespace std;

const std::string PATH_PREFIX = "C:/Users/Bibek Joshi/Desktop/TEST";
const std::string CAPTURED_PREFIX = "C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/";

class EmotionDetector
{
public:
    EmotionDetector()
    {
        curl_global_init(CURL_GLOBAL_ALL); //  application initialization
    }

    ~EmotionDetector()
    {
        curl_global_cleanup(); //  application cleanup
    }

    std::string detect(const cv::Mat &faceImage)
    {
        CURL *curl = curl_easy_init();
        if (!curl)
        {
            std::cerr << "Curl initialization failed." << std::endl;
            return "";
        }

        std::string response_string;
        std::string encoded_image = image_to_base64(faceImage);

        std::string url = "http://localhost:3000/recognize";
        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        json j;
        j["imageBuffer"] = encoded_image;
        std::string postData = j.dump();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);

        return response_string;
    }

private:
    static size_t write_callback(void *contents, size_t size, size_t nmemb, std::string *s)
    {
        size_t newLength = size * nmemb;
        try
        {
            s->append((char *)contents, newLength);
        }
        catch (std::bad_alloc &e)
        {
            // Handle memory problem
            return 0;
        }
        return newLength;
    }

    std::string base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len)
    {
        static const char *base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

        std::string ret;
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];

        while (in_len--)
        {
            char_array_3[i++] = *(bytes_to_encode++);
            if (i == 3)
            {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (i = 0; (i < 4); i++)
                    ret += base64_chars[char_array_4[i]];
                i = 0;
            }
        }

        if (i)
        {
            for (j = i; j < 3; j++)
                char_array_3[j] = '\0';

            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (j = 0; (j < i + 1); j++)
                ret += base64_chars[char_array_4[j]];

            while ((i++ < 3))
                ret += '=';
        }

        return ret;
    }

    std::string image_to_base64(const cv::Mat &img)
    {
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        return base64_encode(enc_msg, buf.size());
    }
};

// Base Abstract Class Feature Detector Definition
class Detector
{
protected:
    CascadeClassifier classifier;

public:
    Detector(const string &classifierPath)
    {
        if (!classifier.load(classifierPath))
        {
            std::cout << "Error loading classifier from " << classifierPath << std::endl;
        }
    }

    // Pure virtual function for detection
    virtual void detect(const Mat &img) = 0;
};

// Face Detection Class Definition
class FaceDetector : public Detector
{
private:
    std::vector<cv::Rect> lastDetectedFaces;

public:
    FaceDetector(const std::string &classifierPath) : Detector(classifierPath){};

    void detect(const Mat &img) override
    {
        vector<Rect> faces;
        classifier.detectMultiScale(img, faces, 1.3, 5);
        for (const auto &face : faces)
        {
            rectangle(img, face.tl(), face.br(), Scalar(50, 50, 255), 2);
        }
        cout << faces.size() << " face(s) found." << endl;
        lastDetectedFaces = faces;
    }

    const std::vector<cv::Rect> &getLastDetectedFaces() const
    {
        return lastDetectedFaces;
    }
};

// Eyes Detection Class Definition
class EyesDetector : public Detector
{
public:
    EyesDetector(const std::string &classifierPath) : Detector(classifierPath){};

    void detect(const Mat &img) override
    {
        //  an empty implementation
    }

    // This method is specific to EyesDetector and does not override the base class's detect method.
    void detect(const Mat &img, const Rect &faceRegion)
    {
        Mat ROI = img(faceRegion);
        vector<Rect> eyes;
        classifier.detectMultiScale(ROI, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (const auto &e : eyes)
        {
            Rect eyeRect(faceRegion.x + e.x, faceRegion.y + e.y, e.width, e.height);
            rectangle(img, eyeRect.tl(), eyeRect.br(), Scalar(0, 255, 0), 2);
        }
    }
};

// Smile Detection Class Definition
class SmileDetector : public Detector
{
public:
    SmileDetector(const std::string &classifierPath) : Detector(classifierPath){};

    void detect(const Mat &img) override
    {
        //  an empty implementation
    }

    // This method is specific to SmileDetector and does not override the base class's detect method.
    void detect(const Mat &img, const Rect &faceRegion)
    {
        Mat ROI = img(faceRegion);
        vector<Rect> smiles;
        classifier.detectMultiScale(ROI, smiles, 1.7, 20, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (const auto &smile : smiles)
        {
            Point center(faceRegion.x + smile.x + smile.width / 2, faceRegion.y + smile.y + smile.height / 2);
            ellipse(img, center, Size(smile.width / 2, smile.height / 2), 0, 0, 360, Scalar(255, 0, 0), 4);
        }
    }
};

// Displaying Detected Image Emotion
void annotateAndDisplayImage(cv::Mat &image, const std::string &emotion, double score)
{
    const std::string emWin = "Emotion";
    cv::Mat displayImage;
    double scaleFactor = 1.0; // Adjust based on your needs
    cv::resize(image, displayImage, cv::Size(), scaleFactor, scaleFactor);

    // Annotate the resized image with the detected emotion and score.
    cv::putText(displayImage, "Detected Emotion: " + emotion, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(displayImage, "Score: " + std::to_string(score), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    // Display the annotated image.
    cv::imshow(emWin, displayImage);

    // Wait for ESC key (27) before closing.
    while (true)
    {
        char key = cv::waitKey(1);
        if (key == 27 || cv::getWindowProperty(emWin, cv::WND_PROP_VISIBLE) < 1)
        {
            break;
        }
    }

    cv::destroyWindow(emWin); // Explicitly destroy the window.
}

// Video Handler Class Definition
class VideoHandler
{
private:
    static VideoHandler *instance;
    cv::VideoCapture video;
    std::string windowName;
    bool captureImage;

    static void onMouse(int event, int x, int y, int flags, void *userdata)
    {
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            instance->captureImage = true;
        }
    }

    // Image Capturing and Saving
    void captureAndSaveImage(const cv::Mat &frame)
    {
        std::time_t now = std::time(nullptr);
        char timestamp[20];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", std::localtime(&now));
        std::string filename = CAPTURED_PREFIX + "captured_" + std::string(timestamp) + ".jpg";

        // Save frame to file
        cv::imwrite(filename, frame);
        std::cout << "Image saved as " << filename << std::endl;
    }

public:
    // Include detector instances
    FaceDetector faceDetector;
    EyesDetector eyesDetector;
    SmileDetector smileDetector;
    EmotionDetector emotionDetector;

    // Video Handler Constructor
    VideoHandler(const std::string &windowName, const std::string &faceCascadePath,
                 const std::string &eyesCascadePath, const std::string &smileCascadePath)
        : windowName(windowName), captureImage(false),
          faceDetector(faceCascadePath),
          eyesDetector(eyesCascadePath),
          smileDetector(smileCascadePath)
    {
        instance = this;
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::setMouseCallback(windowName, onMouse, nullptr);
    }

    // Video Handler Destructor
    ~VideoHandler()
    {
        cv::destroyAllWindows();
    }

    // function to initialize camera
    bool initializeCamera(int cameraIndex)
    {
        video.open(cameraIndex);
        if (!video.isOpened())
        {
            std::cerr << "Error: Couldn't open the camera." << std::endl;
            return false;
        }
        return true;
    }

    // function to capture the video (Video Loop Frame)
    void captureLoop()
    {
        cv::Mat frame;
        while (true)
        {
            video >> frame;
            if (frame.empty())
            {
                std::cerr << "Error: Couldn't read frame from the camera." << std::endl;
                break;
            }

            if (captureImage)
            {
                captureAndSaveImage(frame);
                captureImage = false; // Reset flag

                try
                {
                    std::string emotionResponse = emotionDetector.detect(frame); // Detect emotion.
                    std::cout << "Emotion Response: " << emotionResponse << std::endl;

                    // Assuming detect() returns a JSON string response, parse it.
                    json jsonResponse = json::parse(emotionResponse);

                    // Extract detected emotion and score from the response.
                    std::string detectedEmotion = jsonResponse["emotion"];
                    double emotionScore = jsonResponse["score"];

                    // Optionally, annotate the frame with the detected emotion before displaying it.
                    annotateAndDisplayImage(frame, detectedEmotion, emotionScore);
                }
                catch (const std::exception &e)
                {
                    // Handle exceptions, which could be due to network errors, parsing errors, etc.
                    std::cerr << "Error detecting emotion: " << e.what() << std::endl;

                    // Fallback behavior or error handling.
                    annotateAndDisplayImage(frame, "Failed", 0.0);
                }
            }

            // Detect faces
            faceDetector.detect(frame);
            const auto &detectedFaces = faceDetector.getLastDetectedFaces();

            // Detect eyes and smiles within each detected face region
            for (const auto &face : detectedFaces)
            {
                eyesDetector.detect(frame, face);
                smileDetector.detect(frame, face);
            }

            cv::imshow(windowName, frame);
            char key = cv::waitKey(1);
            if (key == 27 || cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
            {
                break;
            }
        }
    }
};

// static member initialization
VideoHandler *VideoHandler::instance = nullptr;

// Main Function Declaration
int main()
{
    const std::string faceCascadePath = PATH_PREFIX + "/data/haarcascades/lbpcascade_frontalface_improved.xml";
    const std::string eyesCascadePath = PATH_PREFIX + "/data/haarcascades/haarcascade_eye.xml";
    const std::string smileCascadePath = PATH_PREFIX + "/data/haarcascades/haarcascade_smile.xml";

    // VideoHandle Instance Creation
    VideoHandler videoHandler("Camera", faceCascadePath, eyesCascadePath, smileCascadePath);
    if (!videoHandler.initializeCamera(0))
    {
        return -1;
    }

    videoHandler.captureLoop();
    return 0;
}
