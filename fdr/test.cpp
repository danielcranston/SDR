#include <sys/stat.h>
#include <chrono>
#include "ArgsParser.h"
#include "Evaluator.h"
#include "FastDR.h"
#include "Utilities.hpp"
#include <opencv2/opencv.hpp>

bool loadData(const std::string inputDir, cv::Mat& im0, cv::Mat& disp_WTA, cv::Mat& dispGT, cv::Mat& nonocc, Calib&calib)
{
    if (calib.ndisp <= 0)
        printf("Try to retrieve ndisp from file [calib.txt].\n");
    calib = Calib(inputDir + "calib.txt");
    if (calib.ndisp <= 0) {
        printf("ndisp is not speficied.\n");
        return false;
    }

    im0 = cv::imread(inputDir + "im0.png");
    if (im0.empty()) {
        printf("Image im0.png not found in\n");
        printf("%s\n", inputDir.c_str());
        return false;
    }

    disp_WTA = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
    if (disp_WTA.empty()) {
        printf("disp_WTA.pfm is empty.\n");
        return -1;
    }

    dispGT = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
    if (dispGT.empty())
        dispGT = cv::Mat_<float>::zeros(im0.size());

    nonocc = cv::imread(inputDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);
    if (!nonocc.empty())
        nonocc = nonocc == 255;
    else
        nonocc = cv::Mat_<uchar>(im0.size(), 255);

    return true;
}

void LiuDataset(const std::string inputDir, const std::string outputDir, const Options& options)
{
    Parameters params = options.params;

    cv::Mat im0, disp_WTA, dispGT, nonocc;
    Calib calib;

    calib.ndisp = options.ndisp;
    if (!loadData(inputDir, im0, disp_WTA, dispGT, nonocc, calib))
        return;


    printf("ndisp = %d\n", calib.ndisp);
    
    int maxdisp = calib.ndisp;
    double errorThresh = 1.0;
    if (cvutils::contains(inputDir, "trainingQ") || cvutils::contains(inputDir, "testQ"))
        errorThresh = errorThresh / 2.0;
    else if (cvutils::contains(inputDir, "trainingF") || cvutils::contains(inputDir, "testF"))
        errorThresh = errorThresh * 2.0;

    {
        mkdir((outputDir + "debug").c_str(), 0755);

        Evaluator* eval = new Evaluator(dispGT, nonocc, "result", outputDir + "debug/");
        eval->setErrorThreshold(errorThresh);
        eval->start();

        FastDR fdr(im0, disp_WTA, params, maxdisp, 0);

        cv::Mat labeling, refined_disp;
        printf("%s\n", "Begin fdr.run");
        fdr.run(labeling, refined_disp);

        cvutils::io::save_pfm_file("labeling.pfm", labeling);

        cvutils::io::save_pfm_file(outputDir + "disp0FDR.pfm",
                                   refined_disp);

        {
            FILE* fp = fopen((outputDir + "timeFDR.txt").c_str(), "w");
            if (fp != nullptr) {
                fprintf(fp, "%lf\n", eval->getCurrentTime());
                fclose(fp);
            }
        }

        printf("%s\n", "Saved output.");
        if (cvutils::contains(inputDir, "training"))
            eval->evaluate(refined_disp, true, true);

        delete eval;
    }
}


int main(int argc, const char** args) {
    std::cout << "----------- parameter settings -----------" << std::endl;
    ArgsParser parser(argc, args);
    Options options;

    options.loadOptionValues(parser); // lambda, seg_k, inlier_ratio

    mkdir((options.outputDir).c_str(), 0755);

    printf("Running by Middlebury V3 mode.\n");
    LiuDataset(options.targetDir + "/", options.outputDir + "/", options);
    return 0;
}
