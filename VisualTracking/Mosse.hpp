// Mosse Filter Implementation for visual object tracking
// G = F ⋅ (H^*) --> Frequency Domain
// Convolution theorem states that correlation becomes an element-wise multiplication in Fourier Domain
// F -> FFT of the reference object patch
// G -> The goal "response", we'd like to have as a good correlation
//   -> g() is usually chosen as 2D Gaussian signal with a peak in the middle
// H -> The filter that we want to find/learn.
//   -> Under perfect tracking, since F ~= H, G should have a peak in the middle.

#include "Tracker.h"

namespace
{
constexpr double EPS        = 0.00001; // epsilon to avoid division by 0 in normalization
constexpr double LEARN_RATE = 0.75;    // learning rate for the model update
constexpr double PSR_THRESH = 5.7;     // Peak-to-sidelobe ratio threshold for acceptable tracking
} // namespace

class Mosse : public VisualTracker
{
  public:
    void Init(const cv::Mat &refImg, const cv::Rect &refBbox) override
    {
        G_ = GenerateGaussResponse(refBbox);

        cv::Mat refObjPatch;
        cv::getRectSubPix(refImg, refBbox.size(), (refBbox.tl() + refBbox.br()) / 2.0, refObjPatch);
        cv::Mat refObjPatchGray;
        cv::cvtColor(refObjPatch, refObjPatchGray, cv::COLOR_BGR2GRAY);

        cv::createHanningWindow(cosineWindow_, refBbox.size(), CV_32F);

        PreTrain(refObjPatchGray);

        prevBbox_ = refBbox;
    }

    bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut) override
    {
        // Get the patch based on previous bbox
        cv::Mat imgPatch;
        cv::getRectSubPix(inImg, prevBbox_.size(), (prevBbox_.tl() + prevBbox_.br()) / 2.0, imgPatch);
        cv::Mat imgPatchGray;
        cv::cvtColor(imgPatch, imgPatchGray, cv::COLOR_BGR2GRAY);

        // preprocess the image patch before correlation
        Preprocess(imgPatchGray);

        cv::Point deltaXY;
        if (CorrelateWithPrevBbox(imgPatchGray, deltaXY) < PSR_THRESH)
            return false;

        // Update the current bounding box
        cv::Rect2d curBbox{prevBbox_.x + deltaXY.x, prevBbox_.y + deltaXY.y, prevBbox_.width, prevBbox_.height};

        // Update the filter with the new object patch
        // Get the patch based on previous bbox
        cv::Mat newImgPatch;
        cv::getRectSubPix(inImg, curBbox.size(), (curBbox.tl() + curBbox.br()) / 2.0, newImgPatch);
        cv::Mat newImgPatchGray;
        cv::cvtColor(newImgPatch, newImgPatchGray, cv::COLOR_BGR2GRAY);
        Preprocess(newImgPatchGray);

        // new state for A and B
        cv::Mat F_i, A_i, B_i;
        cv::dft(newImgPatchGray, F_i, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(G_, F_i, A_i, 0, true);
        cv::mulSpectrums(F_i, F_i, B_i, 0, true);

        // update A ,B, and H
        A_ = A_ * (1 - LEARN_RATE) + A_i * LEARN_RATE;
        B_ = B_ * (1 - LEARN_RATE) + B_i * LEARN_RATE;
        H_ = divideDFTs(A_, B_);

        prevBbox_ = curBbox;
        bboxOut   = curBbox;
        return true;
    }

    //  Element-wise division of complex numbers in src1 and src2
    static cv::Mat divideDFTs(const cv::Mat &src1, const cv::Mat &src2)
    {
        cv::Mat c1[2], c2[2], a1, a2, s1, s2, denom, re, im;

        // split into re and im per src
        cv::split(src1, c1);
        cv::split(src2, c2);

        // (Re2*Re2 + Im2*Im2) = denom
        //   denom is same for both channels
        cv::multiply(c2[0], c2[0], s1);
        cv::multiply(c2[1], c2[1], s2);
        cv::add(s1, s2, denom);

        // (Re1*Re2 + Im1*Im1)/(Re2*Re2 + Im2*Im2) = Re
        cv::multiply(c1[0], c2[0], a1);
        cv::multiply(c1[1], c2[1], a2);
        cv::divide(a1 + a2, denom, re, 1.0);

        // (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
        cv::multiply(c1[1], c2[0], a1);
        cv::multiply(c1[0], c2[1], a2);
        cv::divide(a1 + a2, denom, im, -1.0);

        // Merge Re and Im back into a complex Matrix
        cv::Mat dst, chn[] = {re, im};
        cv::merge(chn, 2, dst);
        return dst;
    }

  private:
    // Correlates the image Patch from current frame (k) against the bounding box
    // obtained at the previous step (k-1). If nothing has changed since [k], we expect to
    // get a perfectly centered Gaussian. If the object of interest has moved, correlating
    // with the previous bounding box would result in a "shifted" Gaussian.
    // This "shift" tells us how much we should move the previous bounding box to track tbe object.
    double CorrelateWithPrevBbox(const cv::Mat &imgPatch, cv::Point &deltaXY)
    {
        cv::Mat F_i, g_i, G_i;
        cv::dft(imgPatch, F_i, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(F_i, H_, G_i, 0, true); // response of (current image patch)*(Previous_bbox) = g_i
        cv::idft(G_i, g_i, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        cv::imshow("g_i", g_i);
        // Ideally, g_i should have some peak (like a noisy Gaussian) if the object is tracked
        // Find the position of the peak
        double    peakVal;
        cv::Point peakLoc;
        cv::minMaxLoc(g_i, 0, &peakVal, 0, &peakLoc);
        // cv::Point2d prevCenter{(prevBbox_.br() - prevBbox_.tl()) / 2.0};
        // deltaXY.x = peakLoc.x - static_cast<int>(prevCenter.x);
        // deltaXY.y = peakLoc.y - static_cast<int>(prevCenter.y);
        deltaXY.x = peakLoc.x - static_cast<int>(g_i.size().width / 2);
        deltaXY.y = peakLoc.y - static_cast<int>(g_i.size().height / 2);

        std::cout << "delta x y " << deltaXY.x << " " << deltaXY.y << std::endl;

        // normalize response
        cv::Scalar mean, stdDev;
        cv::meanStdDev(g_i, mean, stdDev);
        double PSR{(peakVal - mean[0]) / (stdDev[0] + EPS)}; // peak-to-sidelobe ratio
        return PSR;
    }

    //   Generate a Gaussian signal and return the FFT of it based on the input bounding box
    cv::Mat GenerateGaussResponse(const cv::Rect2d &refBbox)
    {
        cv::Mat g                                          = cv::Mat::zeros(refBbox.height, refBbox.width, CV_32F);
        g.at<float>(refBbox.height / 2, refBbox.width / 2) = 1; // Peak at the middle
        cv::GaussianBlur(g, g, cv::Size(-1, -1), 2.0);
        // Normalize
        double maxVal;
        cv::minMaxLoc(g, 0, &maxVal);
        g = g / maxVal;
        cv::Mat G;
        cv::dft(g, G, cv::DFT_COMPLEX_OUTPUT);
        return G;
    }

    // FFT of the 2D spatial image maps it onto a torus, which causes the edges of the frame to merge.
    // Image is multiplied with a cosine window to gradually reduce the pixels near the edge to 0.
    // Image is log-normalized to deal with low-contrast lighting conditions
    void Preprocess(cv::Mat &inImg)
    {
        inImg.convertTo(inImg, CV_32F);
        cv::log(inImg + 1.0, inImg);
        // Normalize
        cv::Scalar mean, stdDev;
        cv::meanStdDev(inImg, mean, stdDev);
        inImg = ((inImg - mean[0]) / (stdDev[0] + EPS)).mul(cosineWindow_);
    }

    cv::Mat randomWarp(const cv::Mat &inImg) const
    {
        static cv::RNG rng(8031965);

        // random rotation
        double C   = 0.1;
        double ang = rng.uniform(-C, C);
        double c = cos(ang), s = sin(ang);
        // affine warp matrix
        cv::Mat_<float> W(2, 3);
        W << c + rng.uniform(-C, C), -s + rng.uniform(-C, C), 0, s + rng.uniform(-C, C), c + rng.uniform(-C, C), 0;

        // random translation
        cv::Mat_<float> center_warp(2, 1);
        center_warp << inImg.cols / 2, inImg.rows / 2;
        W.col(2) = center_warp - (W.colRange(0, 2)) * center_warp;

        cv::Mat warped;
        cv::warpAffine(inImg, warped, W, inImg.size(), cv::BORDER_REFLECT);
        return warped;
    }

    // Given the reference object patch we want to track and the ideal response G,
    // creates the initial filter H by training on the warped variants of the reference object
    void PreTrain(cv::Mat &refObjPatch)
    {
        cv::Mat origRefObjPatch;
        refObjPatch.copyTo(origRefObjPatch); // f_i

        Preprocess(refObjPatch);
        cv::Mat F_i;
        cv::dft(refObjPatch, F_i, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(G_, F_i, A_, 0, true);
        cv::mulSpectrums(F_i, F_i, B_, 0, true);

        // Warp the image in various ways for training
        for (int i = 0; i < 8; i++)
        {
            cv::Mat warpedPatch = randomWarp(origRefObjPatch);
            Preprocess(warpedPatch);

            cv::Mat F_i, A_i, B_i;
            cv::dft(warpedPatch, F_i, cv::DFT_COMPLEX_OUTPUT);
            cv::mulSpectrums(G_, F_i, A_i, 0, true);
            cv::mulSpectrums(F_i, F_i, B_i, 0, true);
            A_ += A_i;
            B_ += B_i;
        }

        H_ = divideDFTs(A_, B_);
    }

    cv::Mat G_;
    cv::Mat A_;
    cv::Mat B_;
    cv::Mat H_;

    cv::Mat cosineWindow_;

    cv::Rect2d prevBbox_;
};