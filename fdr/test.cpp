#include <sys/stat.h>
#include <chrono>
#include "ArgsParser.h"
#include "Evaluator.h"
#include "FastDR.h"
#include "Utilities.hpp"
#include <opencv2/opencv.hpp>


/**
 * Update the invalid disparity (0 value) to a random disparity from
 * (1,max_disp)
 */
void preprocess_disp(cv::Mat& disp_WTA, const int max_disp) {
    cv::RNG rng;
    for (int y = 0; y < disp_WTA.rows; y++)
        for (int x = 0; x < disp_WTA.cols; x++) {
            if (disp_WTA.at<float>(y, x) < 1)
                disp_WTA.at<float>(y, x) = rng.operator() (max_disp - 1) + 1;
        }
}

bool loadData(const std::string path, std::string item, cv::Mat& im0, cv::Mat& disp_WTA, cv::Mat& dispGT, cv::Mat& nonocc, Calib&calib)
{
    // if (calib.ndisp <= 0)
    //     printf("Try to retrieve ndisp from file [calib.txt].\n");
    // calib = Calib(inputDir + "calib.txt");
    // if (calib.ndisp <= 0) {
    //     printf("ndisp is not speficied.\n");
    //     return false;
    // }


    im0 = cv::imread(path + "/left_rgb/" + item + ".png");
    if (im0.empty()) {
        std::cout << "RGB image not found: " << path + "/left_rgb/" + item + ".png" << std::endl;
        return false;
    }

    disp_WTA = cvutils::io::read_pfm_file(path + "/left_initial_disparity/" + item + ".pfm");
    if (disp_WTA.empty()) {
        printf("disp_WTA is empty.\n");
        return -1;
    }
    printf("preprocessing disp...\n");
    double min, max;
    cv::minMaxLoc(disp_WTA, &min, &max);
    printf("  max: %f\n", max);
    preprocess_disp(disp_WTA, max);//calib.ndisp);
    printf("done\n");
    
    dispGT = cvutils::io::read_pfm_file(path + "/left_gt/" + item + ".pfm");
    if (dispGT.empty())
        dispGT = cv::Mat_<float>::zeros(im0.size());


    nonocc = cv::Mat_<uchar>(im0.size(), 255);

    return true;
}

void LiuDataset(const std::string path, const std::string item, const Options& options)
{
    Parameters params = options.params;

    cv::Mat im0, disp_WTA, dispGT, nonocc;
    Calib calib;

    std::cout << "  path is " << path << std::endl;
    std::cout << "  item is " << item << std::endl;

    calib.ndisp = 256;
    if (!loadData(path, item, im0, disp_WTA, dispGT, nonocc, calib))
        return;


    printf("ndisp = %d\n", calib.ndisp);
    
    int maxdisp = calib.ndisp;
    double errorThresh = 1.0;

    {
        // mkdir((outputDir + "debug").c_str(), 0755);

        // Evaluator* eval = new Evaluator(dispGT, nonocc, "result", outputDir + "debug/");
        // eval->setErrorThreshold(errorThresh);
        // eval->start();

        FastDR fdr(im0, disp_WTA, params, maxdisp, 0);

        cv::Mat labeling, refined_disp;
        printf("%s\n", "Begin fdr.run");
        fdr.run(labeling, refined_disp);

        // cvutils::io::save_pfm_file("labeling.pfm", labeling);

        cvutils::io::save_pfm_file(path + "/left_output_sdr/" + item + ".pfm",
                                   refined_disp);

        // {
        //     FILE* fp = fopen((outputDir + "timeFDR.txt").c_str(), "w");
        //     if (fp != nullptr) {
        //         fprintf(fp, "%lf\n", eval->getCurrentTime());
        //         fclose(fp);
        //     }
        // }

        // printf("%s\n", "Saved output.");
        // if (cvutils::contains(inputDir, "training"))
        //     eval->evaluate(refined_disp, true, true);

        // delete eval;
    }
}

int main(int argc, const char** args) {
    std::cout << "----------- parameter settings -----------" << std::endl;
    ArgsParser parser(argc, args);
    Options options;

    options.loadOptionValues(parser); // lambda, seg_k, inlier_ratio

    // mkdir((options.outputDir).c_str(), 0755);

    printf("Running Liu Dataset mode. -------------------------\n");
    LiuDataset(options.targetDir, options.outputDir, options);
    return 0;
}
