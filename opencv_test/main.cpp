#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    VideoCapture cap(0);//"Resources/test_video.mp4");
    Mat img, original_img, face_img;
    high_resolution_clock::time_point start, end;
    int duration;
    CascadeClassifier face_cas, eye_cas;
    face_cas.load("Resources/haarcascade_frontalface_default.xml");
    eye_cas.load("Resources/haarcascade_eye.xml");
    vector<Rect> faces, eyes;
    Rect face;
    float reduction_factor = 0.25;

    while (true) {
        start = high_resolution_clock::now();
        if(cap.read(original_img)) {
            resize(original_img, img, Size(), reduction_factor, reduction_factor);
            putText(img, to_string(duration) + " ms, " + to_string((int)1000.0/duration) + " FPS", Point(0, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
            
            face_cas.detectMultiScale(img, faces, 1.1, 10);

            face = faces[0];
            for (auto tmp_face : faces) {
                if (face.area() < tmp_face.area())
                   face = tmp_face;
                rectangle(img, tmp_face.tl(), tmp_face.br(), Scalar(0, 255, 0));
            }
            //face_img = img(face);

             eye_cas.detectMultiScale(img, eyes, 1.1);

             for (auto eye : eyes) {
                 rectangle(img, eye.tl(), eye.br(), Scalar(0, 0, 255));
             }
        

            imshow("cam", img);
            if (waitKey(1) == 27) //break == esc
                break;
            end = high_resolution_clock::now();
            duration = duration_cast<milliseconds>(end - start).count();
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}