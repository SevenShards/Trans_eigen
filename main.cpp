#include <iostream>

#include <iostream>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <map>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

using namespace cv;
using namespace Eigen;
using namespace std;

struct ReturnVal_DisMat {
    int delta;
    MatrixXd a;

};

int Image_to_array();

int DistilingProgram(MatrixXd pixelChannelMatrix, string channelName, int height, int width);

ReturnVal_DisMat DistilingMatrix(string channelName, int loopNumber, int delta, MatrixXd pixelMatrix);

MatrixXd reconstructMatrix(string channelName, int loopNumber);

MatrixXd readBaseBin(string channelName, int number);

MatrixXd readSignalBin(string channelName, int number);

MatrixXd restorePixelMatrixInHorizontal(MatrixXd restore_pixel_matrix_horizontal_temp, MatrixXd b_reconstructMatrix);

MatrixXd
restorePixelMatrixInVertical(MatrixXd restore_pixel_matrix_temp, MatrixXd restore_pixel_matrix_horizontal_temp);

int Matrix_To_Image();

map<string, MatrixXd> basebuffer;
map<string, MatrixXd> signalbuffer;
map<string, MatrixXd> channelbuffer;

Mat src = imread("../keyframe/Iframe-01.jpeg");

int main() {

    Image_to_array();
    Matrix_To_Image();
    return 0;
}

int Matrix_To_Image() {

    Mat dst;
    dst.create(src.size(), src.type());

    int height = src.rows;
    int width = src.cols;

    map<string, MatrixXd>::iterator it_b;
    it_b = channelbuffer.find("b_restore");

    map<string, MatrixXd>::iterator it_g;
    it_g = channelbuffer.find("g_restore");

    map<string, MatrixXd>::iterator it_r;
    it_r = channelbuffer.find("r_restore");


    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int b = ((*it_b).second)(row, col);
            int g = ((*it_g).second)(row, col);
            int r = ((*it_r).second)(row, col);
            dst.at<Vec3b>(row, col)[0] = b;
            dst.at<Vec3b>(row, col)[1] = g;
            dst.at<Vec3b>(row, col)[2] = r;

        }
    }

    namedWindow("output", CV_WINDOW_AUTOSIZE);
    imshow("output", dst);

    waitKey(0);
    return 0;
}

int Image_to_array() {


    if (src.data == nullptr) {
        cout << "图片不存在" << endl;
        return -1;
    }

    int height = src.rows; //720
    int width = src.cols;  //1280

    MatrixXd r_matrix(height, width);
    MatrixXd g_matrix(height, width);
    MatrixXd b_matrix(height, width);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            int b = src.at<Vec3b>(row, col)[0];
            int g = src.at<Vec3b>(row, col)[1];
            int r = src.at<Vec3b>(row, col)[2];
            //cout << "b: " << b << " g: " << g << " r: " << r << endl;
            b_matrix(row, col) = b;
            g_matrix(row, col) = g;
            r_matrix(row, col) = r;

            //cout << "row: " << row << " col: " << col << " r_matrix: " << r_matrix(row, col) << " g_matrix: "
            //     << g_matrix(row, col) << " b_matrix: " << b_matrix(row, col) << endl;
        }
    }
    //cout << r_matrix << endl;

    DistilingProgram(r_matrix, "r", height, width);
    DistilingProgram(g_matrix, "g", height, width);
    DistilingProgram(b_matrix, "b", height, width);

    //cout << "pause" << endl;
    return 0;
}

int DistilingProgram(MatrixXd pixelChannelMatrix, string channelName, int height, int width) {


    MatrixXd restore_pixel_matrix(720, 1280);
    MatrixXd restore_pixel_matrix_horizontal_temp(720, 16);


    vector<MatrixXd> allMatrix;

    int count_length = 0;
    bool vertical_flag = true;

    //split to 80x(16x720)
    for (int i = 0; i <= 1264; i += 16) {
        int row = pixelChannelMatrix.rows();
        MatrixXd m_temp_vertical(row, 16);
        m_temp_vertical = pixelChannelMatrix.block(0, i, row, 16);

        int count_width = 0;
        bool horizontal_flag = true;

        int count_reconstruct = 0;
        int count_no_reconstruct = 0;
        //45x(16x16)
        for (int j = 0; j <= 704; j += 16) {


            MatrixXd m_temp(16, 16);
            m_temp = m_temp_vertical.block<16, 16>(j, 0);

            int max = m_temp.maxCoeff();
            int min = m_temp.minCoeff();
            int delta = max - min;
            MatrixXd resultMatrix = m_temp;
            if (delta <= 2) {
                count_no_reconstruct ++;
                if (horizontal_flag) {
                    restore_pixel_matrix_horizontal_temp = m_temp;
                    horizontal_flag = false;
                } else {
                    restore_pixel_matrix_horizontal_temp = restorePixelMatrixInHorizontal(
                            restore_pixel_matrix_horizontal_temp, m_temp);
                }
            }
            int loopNumber = 0;
            while (delta > 2) {
                loopNumber += 1;
                ReturnVal_DisMat returnValDisMat = DistilingMatrix(channelName, loopNumber, delta, resultMatrix);
                resultMatrix = returnValDisMat.a;
                delta = returnValDisMat.delta;
                if (delta <= 2) {
                    MatrixXd b_reconstructMatrix = reconstructMatrix(channelName, loopNumber);
                    count_reconstruct++;
                    //cout << "b_reconstructMatrix.rows(): " << b_reconstructMatrix.rows()
                     //    << "   b_reconstructMatrix.cols():" << b_reconstructMatrix.cols() << endl;

                    if (horizontal_flag) {
                        restore_pixel_matrix_horizontal_temp = b_reconstructMatrix;
                        horizontal_flag = false;
                    } else {
                        restore_pixel_matrix_horizontal_temp = restorePixelMatrixInHorizontal(
                                restore_pixel_matrix_horizontal_temp, b_reconstructMatrix);
                    }
                    break;
                }
            }

            count_width += 1;

        }
        //cout << "count_reconstruct: " << count_reconstruct << endl;
        //cout << "count_no_reconstruct: " << count_no_reconstruct << endl;

        //cout << "count_width: " << count_width << endl;
        if (vertical_flag) {
            //cout << " if (vertical_flag)" << endl;
            restore_pixel_matrix = restore_pixel_matrix_horizontal_temp;
            vertical_flag = false;
        } else {
            //cout << "else" << endl;
            restore_pixel_matrix = restorePixelMatrixInVertical(restore_pixel_matrix,
                                                                restore_pixel_matrix_horizontal_temp);
        }
        count_length += 1;
        //cout << "count_length" << count_length << endl;

    }
    cout << "count_length" << count_length << endl;

    map<string, MatrixXd>::iterator it_channel;
    it_channel = channelbuffer.find(channelName + "_restore");
    if (it_channel != basebuffer.end()) {
        channelbuffer.erase(it_channel);
    }
    channelbuffer.insert({channelName + "_restore", restore_pixel_matrix});

    return 0;
}

ReturnVal_DisMat DistilingMatrix(string channelName, int loopNumber, int delta, MatrixXd pixelMatrix) {


    int max = pixelMatrix.maxCoeff();
    int min = pixelMatrix.minCoeff();
    delta = max - min;
    int mid = 0;
    if (delta % 2 != 0) {
        mid = (max + min + 1) / 2;
    } else {
        mid = (max + min) / 2;
    }
    MatrixXd b = MatrixXd::Ones(pixelMatrix.rows(), pixelMatrix.cols()) * mid;

    map<string, MatrixXd>::iterator it_base;
    it_base = basebuffer.find(channelName + "_b" + to_string(loopNumber));
    if (it_base != basebuffer.end()) {
        basebuffer.erase(it_base);
    }
    basebuffer.insert({channelName + "_b" + to_string(loopNumber), b});
    //cout << "channelName+\"_b\"+to_string(flagNumber): " << channelName + "_b" + to_string(loopNumber) << endl;

    //MatrixXd d = (pixelMatrix - MatrixXd::Ones(d.rows(), d.cols()) * mid);
    MatrixXd d = pixelMatrix - b;

    MatrixXd a = MatrixXd::Zero(pixelMatrix.rows(), pixelMatrix.cols());
    MatrixXd s = MatrixXd::Zero(pixelMatrix.rows(), pixelMatrix.cols());

    for (int i = 0; i < d.rows(); i++) {
        for (int j = 0; j < d.cols(); j++) {
            if (d(i, j) < 0) {
                a(i, j) = d(i, j) * (-1);
                s(i, j) = -1;
            } else {
                a(i, j) = d(i, j);
                s(i, j) = 1;
            }
        }
    }

    map<string, MatrixXd>::iterator it_signal;
    it_signal = signalbuffer.find(channelName + "_s" + to_string(loopNumber));
    if (it_signal != signalbuffer.end()) {
        signalbuffer.erase(it_signal);
    }
    signalbuffer.insert({channelName + "_s" + to_string(loopNumber), s});
    //cout << "channelName+\"_s\"+to_string(flagNumber): " << channelName + "_s" + to_string(loopNumber) << endl;

    ReturnVal_DisMat returnValDisMat = {delta, a};

    return returnValDisMat;
}

MatrixXd reconstructMatrix(string channelName, int loopNumber) {

    MatrixXd b_reconstructMatrix;

    map<string, MatrixXd>::iterator it;
    it = basebuffer.find(channelName + "_b1");
    if (it != basebuffer.end()) {
        b_reconstructMatrix = (*it).second;
    }

    for (int i = loopNumber; i > 0; i--) {
        if (i == 1) {
            break;
        }
        MatrixXd b_temp = readBaseBin(channelName, i);
        b_reconstructMatrix = b_reconstructMatrix + b_temp;
    }
    return b_reconstructMatrix;

}

MatrixXd readBaseBin(string channelName, int number) {
    MatrixXd b;

    map<string, MatrixXd>::iterator it;
    it = basebuffer.find(channelName + "_b" + to_string(number));
    if (it != basebuffer.end()) {
        b = (*it).second;
    }

    for (int i = number; i > 0; i--) {
        if (i == 1) {
            break;
        }
        MatrixXd s = readSignalBin(channelName, i);
        b = b.array() * s.array();
    }
    return b;
}

MatrixXd readSignalBin(string channelName, int number) {
    MatrixXd s;

    map<string, MatrixXd>::iterator it;
    it = signalbuffer.find(channelName + "_s" + to_string(number));
    if (it != signalbuffer.end()) {
        s = (*it).second;
    }
    return s;
}

MatrixXd restorePixelMatrixInHorizontal(MatrixXd restore_pixel_matrix_horizontal_temp, MatrixXd b_reconstructMatrix) {

    MatrixXd restore_pixel_matrix_horizontal(restore_pixel_matrix_horizontal_temp.rows() + 16, 16);
    restore_pixel_matrix_horizontal << restore_pixel_matrix_horizontal_temp, b_reconstructMatrix;
    return restore_pixel_matrix_horizontal;

}

MatrixXd
restorePixelMatrixInVertical(MatrixXd restore_pixel_matrix_temp, MatrixXd restore_pixel_matrix_horizontal_temp) {

    MatrixXd restore_pixel_matrix_vertical_temp(720, restore_pixel_matrix_temp.cols() + 16);

    restore_pixel_matrix_vertical_temp << restore_pixel_matrix_temp, restore_pixel_matrix_horizontal_temp;
    /**
    cout << "restore_pixel_matrix_temp.rows(): " << restore_pixel_matrix_temp.rows()
         << "   restore_pixel_matrix_temp.cols():" << restore_pixel_matrix_temp.cols() << endl;
    cout << "restore_pixel_matrix_horizontal_temp.rows(): " << restore_pixel_matrix_horizontal_temp.rows()
         << "   restore_pixel_matrix_horizontal_temp():" << restore_pixel_matrix_horizontal_temp.cols() << endl;
    cout << "rrestore_pixel_matrix_vertical_temp.rows(): " << restore_pixel_matrix_vertical_temp.rows()
         << "   restore_pixel_matrix_vertical_temp():" << restore_pixel_matrix_vertical_temp.cols() << endl;
   **/

    return restore_pixel_matrix_vertical_temp;
}
