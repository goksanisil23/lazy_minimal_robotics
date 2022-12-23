#include "Homography.h"

namespace homography
{
Homography::Homography()
{
    cX_ = 512.0f;
    cY_ = 320.0f;
    fX_ = 512.0f;
    fY_ = 512.0f;

    K_ = (cv::Mat_<float>(3, 3) << fX_, 0.0, cX_, 0.0, fY_, cY_, 0.0, 0.0, 1.0);

    orb_           = cv::ORB::create(homography::MAX_ORB_FEAUTURES);
    flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

// This function stitches the right image to the left image through homography
void Homography::StitchRightToLeft(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    static int imgIdx = 0;
    // 1) Extract features & descriptors
    cv::Mat leftImgGray, rightImgGray;
    cv::cvtColor(leftImg, leftImgGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImg, rightImgGray, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    std::vector<cv::Point2d>  goodkeypointsLeft, goodkeypointsRight;
    cv::Mat                   descriptorsLeft, descriptorsRight; // (num_keypoints x descriptor_size) = (3000 x 32)
    orb_->detectAndCompute(leftImgGray, cv::Mat(), keypointsLeft, descriptorsLeft);
    orb_->detectAndCompute(rightImgGray, cv::Mat(), keypointsRight, descriptorsRight);
    descriptorsLeft.convertTo(descriptorsLeft, CV_32F);   // descriptors are integers but for knn we neet float
    descriptorsRight.convertTo(descriptorsRight, CV_32F); // descriptors are integers but for knn we neet float

    // 2) Match against previous frame features
    std::vector<std::vector<cv::DMatch>> knnMatches;
    std::vector<cv::DMatch>              goodMatches;

    if (!descriptorsLeft.empty() && !descriptorsRight.empty())
    {
        // find 2 best matches, with distance in increasing order
        flann_matcher_->knnMatch(descriptorsLeft, descriptorsRight, knnMatches, 2);
        for (auto el : knnMatches)
        {
            // Take the better match, only if it's considerably more dominant than
            // the next best match (considerably smaller distance)
            if (el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
            {
                goodMatches.push_back(el[0]);
                goodkeypointsLeft.push_back(keypointsLeft.at(el[0].queryIdx).pt);
                goodkeypointsRight.push_back(keypointsRight.at(el[0].trainIdx).pt);
            }
        }
    }

    // Show the matches
    {
        cv::Mat img_matches;
        // only draw the good matches
        cv::drawMatches(leftImgGray,
                        keypointsLeft,
                        rightImgGray,
                        keypointsRight,
                        goodMatches,
                        img_matches,
                        cv::Scalar::all(-1),
                        cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("matches", img_matches);
        cv::moveWindow("matches", 30, 500);
        cv::waitKey(1000);
    }
    if (!(goodMatches.size() > MIN_MATCHES))
    {
        std::cerr << "insufficient matches: " << goodMatches.size() << std::endl;
        return;
    }

    // Use good matched keypoints from both images for homography computation
    // will transform goodkeypointsRight --> goodkeypointsLeft
    auto homographyMatrix = cv::findHomography(goodkeypointsRight, goodkeypointsLeft, cv::RANSAC, RANSAC_ERR_THRES);

    std::cout << "homography:" << std::endl << homographyMatrix << std::endl;

    // Bounding box of the 2nd image that will be warped
    // (3x4)
    // |u_top_left u_top_right u_bottom_left u_bottom_right|
    // |v_top_left v_top_right v_bottom_left v_bottom_right|
    // |     1          1            1             1       |
    cv::Mat bboxRightImg = (cv::Mat_<double>(3, 4) << 0.0,
                            rightImg.cols - 1.0,
                            0.0,
                            rightImg.cols - 1.0,
                            0.0,
                            0.0,
                            rightImg.rows - 1.0,
                            rightImg.rows - 1.0,
                            1,
                            1,
                            1,
                            1);

    // Get the bounding box coordinates after homography transformation
    cv::Mat bboxRightImgWarped = homographyMatrix * bboxRightImg;
    // Scale back to [u,v,1]
    cv::Mat uValsBboxRightWarped = bboxRightImgWarped(cv::Rect(0, 0, bboxRightImgWarped.cols, 1));
    cv::Mat vValsBboxRightWarped = bboxRightImgWarped(cv::Rect(0, 1, bboxRightImgWarped.cols, 1));
    cv::Mat scaleBbox1Warped     = bboxRightImgWarped(cv::Rect(0, 2, bboxRightImgWarped.cols, 1));
    uValsBboxRightWarped         = uValsBboxRightWarped.mul(1 / scaleBbox1Warped);
    vValsBboxRightWarped         = vValsBboxRightWarped.mul(1 / scaleBbox1Warped);

    // If the bbox of the morphed image has (-) coordinates, they will be clipped.
    // If the bbox of the morphed image larger than original image size, they will be clipped.
    // Hence, we extend the bbox accordingly
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(uValsBboxRightWarped, &minU, &maxU);
    cv::minMaxLoc(vValsBboxRightWarped, &minV, &maxV);
    minU = std::round(minU);
    minV = std::round(minV);
    maxU = std::round(maxU);
    maxV = std::round(maxV);
    cv::Rect warpedImgRightBbox;
    warpedImgRightBbox.x      = minU;
    warpedImgRightBbox.y      = minV;
    warpedImgRightBbox.width  = maxU - minU;
    warpedImgRightBbox.height = maxV - minV;

    int extensionU = 0;
    int extensionV = 0;
    if (minU < 0)
    {
        warpedImgRightBbox.x += std::abs(minU);
        extensionU = std::abs(minU);
    }
    if (minV < 0)
    {
        warpedImgRightBbox.y += std::abs(minV);
        extensionV = std::abs(minV);
    }

    std::cout << "warped img w: " << warpedImgRightBbox.width << " warped img h: " << warpedImgRightBbox.height
              << std::endl;
    std::cout << "warped u0: " << warpedImgRightBbox.x << " warped v0: " << warpedImgRightBbox.y << std::endl;

    // Get the total bounding box of (warpedImgRightBbox  + img0Bbox)
    cv::Rect img0Bbox(0, 0, leftImg.cols, leftImg.rows);
    int      stitch_x_min, stitch_x_max, stitch_y_min, stitch_y_max;
    stitch_x_min = std::min(img0Bbox.x, warpedImgRightBbox.x);
    stitch_y_min = std::min(img0Bbox.y, warpedImgRightBbox.y);
    stitch_x_max = std::max(img0Bbox.x + img0Bbox.width, warpedImgRightBbox.x + warpedImgRightBbox.width);
    stitch_y_max = std::max(img0Bbox.y + img0Bbox.height, warpedImgRightBbox.y + warpedImgRightBbox.height);

    // Shift (translation) matrix:
    // |1 0 tu|
    // |0 1 tv|
    // |0 0  1|
    cv::Mat shift                   = (cv::Mat_<double>(3, 3) << 1.0, 0.0, extensionU, 0, 1, extensionV, 0, 0, 1);
    cv::Mat shiftedHomographyMatrix = shift * homographyMatrix;
    std::cout << "shifted homography:" << std::endl << shiftedHomographyMatrix << std::endl;

    cv::Mat stitched;
    cv::warpPerspective(rightImg,
                        stitched,
                        shiftedHomographyMatrix,
                        cv::Size(stitch_x_max - stitch_x_min, stitch_y_max - stitch_y_min));
    leftImg.copyTo(stitched(cv::Rect(extensionU, extensionV, leftImg.cols, leftImg.rows)));

    cv::imshow("stitched", stitched);
    cv::waitKey(0);
}

cv::Mat Homography::FindHomography(const std::vector<cv::Point2d> &keypointsLeft,
                                   const std::vector<cv::Point2d> &keypointsRight)
{
    (void)keypointsLeft;
    (void)keypointsRight;
}

} // namespace homography