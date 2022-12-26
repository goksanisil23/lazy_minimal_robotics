#include "Homography.h"

namespace homography
{

Homography::Homography(const CAMERA_TYPE &cameraType) : planeReference_{nullptr}
{
    switch (cameraType)
    {
    case (CAMERA_TYPE::OAK_D_RIGHT):
    {
        cX_ = 627.2868042f;
        cY_ = 354.63162231f;
        fX_ = 803.40545654f;
        fY_ = 803.40545654f;

        K_ = (cv::Mat_<float>(3, 3) << fX_, 0.0, cX_, 0.0, fY_, cY_, 0.0, 0.0, 1.0);
        cv::cv2eigen(K_, KEigen_);

        distCoeffs_ = (cv::Mat_<double>(8, 1) << -10.612578392028809,
                       96.1052474975586,
                       -0.0003611376159824431,
                       4.256470128893852e-05,
                       -75.63127899169922,
                       -10.658458709716797,
                       95.71613311767578,
                       -74.4359359741211);

        break;
    }
    case (CAMERA_TYPE::CARLA_1024_640_PINHOLE):
    {
        cX_ = 512.0f;
        cY_ = 320.0f;
        fX_ = 512.0f;
        fY_ = 512.0f;

        K_ = (cv::Mat_<float>(3, 3) << fX_, 0.0, cX_, 0.0, fY_, cY_, 0.0, 0.0, 1.0);
        break;
    }
    }

    orb_           = cv::ORB::create(homography::MAX_FEAUTURES);
    sift_          = cv::SIFT::create(homography::MAX_FEAUTURES);
    flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

cv::Mat Homography::UndistortImage(const cv::Mat &inputImg)
{
    cv::Mat undistorted;
    cv::undistort(inputImg, undistorted, K_, distCoeffs_);
    return undistorted;
}

void Homography::SetPlaneReferenceImage(const cv::Mat &planeReferenceImg,
                                        const double  &planeReferenceObjWidth,
                                        const double  &planeReferenceObjHeight)
{
    planeReference_ =
        std::make_unique<PlaneReference>(planeReferenceImg, planeReferenceObjWidth, planeReferenceObjHeight);
    planeReference_->GenerateFeatureDescriptors(sift_);
}

void Homography::FindMatches(const cv::Mat &dstImg, const cv::Mat &srcImg)
{
    std::vector<cv::KeyPoint> keypointsSrc;
    cv::Mat                   descriptorsSrc;
    cv::Mat                   srcImgGray;

    // Keypoints and descriptors of the reference image is already computed
    if (planeReference_ != nullptr)
    {
        keypointsSrc   = planeReference_->keypoints_;
        descriptorsSrc = planeReference_->descriptors_;
        srcImgGray     = planeReference_->referenceImgGray_;
    }
    else
    {
        cv::cvtColor(srcImg, srcImgGray, cv::COLOR_RGB2GRAY);
        sift_->detectAndCompute(srcImgGray, cv::Mat(), keypointsSrc, descriptorsSrc);
        descriptorsSrc.convertTo(descriptorsSrc, CV_32F); // descriptors are integers but for knn we neet float
    }

    goodKeypointsSrc_.clear();
    goodKeypointsDst_.clear();
    goodKeypointsSrcIdxs_.clear();

    static int imgIdx = 0;
    // 1) Extract features & descriptors
    cv::Mat dstImgGray;
    cv::cvtColor(dstImg, dstImgGray, cv::COLOR_RGB2GRAY);
    std::vector<cv::KeyPoint> keypointsDst;
    cv::Mat                   descriptorsDst; // (num_keypoints x descriptor_size) = (3000 x 32)
    // orb_->detectAndCompute(dstImgGray, cv::Mat(), keypointsDst, descriptorsDst);
    // orb_->detectAndCompute(srcImgGray, cv::Mat(), keypointsSrc, descriptorsSrc);
    sift_->detectAndCompute(dstImgGray, cv::Mat(), keypointsDst, descriptorsDst);

    descriptorsDst.convertTo(descriptorsDst, CV_32F); // descriptors are integers but for knn we neet float

    // 2) Match against previous frame features
    std::vector<std::vector<cv::DMatch>> knnMatches;
    std::vector<cv::DMatch>              goodMatches;

    if (!descriptorsDst.empty() && !descriptorsSrc.empty())
    {
        // find 2 best matches, with distance in increasing order
        flann_matcher_->knnMatch(descriptorsDst, descriptorsSrc, knnMatches, 2);
        for (auto el : knnMatches)
        {
            // Take the better match, only if it's considerably more dominant than
            // the next best match (considerably smaller distance)
            if (el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
            {
                goodMatches.push_back(el[0]);
                goodKeypointsDst_.push_back(keypointsDst.at(el[0].queryIdx).pt);
                goodKeypointsSrc_.push_back(keypointsSrc.at(el[0].trainIdx).pt);
                goodKeypointsSrcIdxs_.push_back(el[0].trainIdx);
            }
        }
    }

    // Show the matches
    {
        cv::Mat img_matches;
        // only draw the good matches
        cv::drawMatches(dstImgGray,
                        keypointsDst,
                        srcImgGray,
                        keypointsSrc,
                        goodMatches,
                        img_matches,
                        cv::Scalar::all(-1),
                        cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("matches", img_matches);
        cv::moveWindow("matches", 30, 500);
        cv::waitKey(10);
    }
    if (!(goodMatches.size() > MIN_MATCHES))
    {
        std::cerr << "insufficient matches: " << goodMatches.size() << std::endl;
    }
    else
    {
        std::cout << "good matches: " << goodMatches.size() << std::endl;
    }
}

// Computes the homography that warps srcImg onto dstImg
cv::Mat Homography::ComputeHomography(const HOMOGRAPHY_MODE &homMode)
{
    if (goodKeypointsDst_.size() > MIN_MATCHES)
    {
        switch (homMode)
        {
        case (HOMOGRAPHY_MODE::IMAGE_TO_IMAGE):
        {
            auto homographyMatrix =
                cv::findHomography(goodKeypointsSrc_, goodKeypointsDst_, cv::RANSAC, RANSAC_ERR_THRES);
            // auto homographyMatrix = cv::findHomography(goodKeypointsSrc_, goodKeypointsDst_, cv::RHO);
            return homographyMatrix;
        }
        case (HOMOGRAPHY_MODE::OBJ_TO_IMAGE):
        {
            // Use world points in homography calculation
            // Pick the world points based on well matched feature indices
            std::vector<cv::Point3d> goodWorldPoints;
            for (const auto &goodKpIdx : goodKeypointsSrcIdxs_)
            {
                goodWorldPoints.emplace_back(planeReference_->worldPoints_.at(goodKpIdx));
            }

            auto homographyMatrix =
                cv::findHomography(goodWorldPoints, goodKeypointsDst_, cv::RANSAC, RANSAC_ERR_THRES);
            // auto homographyMatrix = cv::findHomography(goodKeypointsSrc_, goodKeypointsDst_, cv::RHO);
            return homographyMatrix;
        }
        default:
            return cv::Mat();
            break;
        }
    }
    else
    {
        std::cout << "Skipping homography calculation, insufficient matches\n";
        return cv::Mat();
    }
}

Eigen::Matrix4d Homography::ComputeCameraPose(const cv::Mat &homographyMatrix3d)
{
    // Convert the homography to Eigen
    Eigen::Matrix3d homographyMatrixEig;
    cv::cv2eigen(homographyMatrix3d, homographyMatrixEig);
    Eigen::Matrix3d M = KEigen_.inverse() * homographyMatrixEig;
    // Left 2 columns of M gives R1_hat and R2_hat
    Eigen::MatrixXd R1R2_hat = M.leftCols<2>();
    // Perform SVD since R1R2 is not guaranteed to be orthogonal.
    auto svd = R1R2_hat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Compute R1R2 (the two first columns of R) from the result of the SVD.
    Eigen::Matrix<double, 3, 2> R1R2 = svd.matrixU() * svd.matrixV().transpose();
    // Construct R3 by cross product of R1 & R2 (since they're orthogonal after SVD)
    // Check det(R)!
    Eigen::Matrix3d R;
    R.leftCols<2>() = R1R2;
    R.col(2)        = R1R2.col(0).cross(R1R2.col(1));

    if (R.determinant() < 0)
    {
        R.col(2) *= -1.0;
    }

    // Compute the scale factor lambda.
    double lambda = (R1R2.array() * R1R2_hat.array()).sum() / (R1R2_hat.array() * R1R2_hat.array()).sum();

    // Extract the translation t.
    Eigen::Vector3d t = M.col(2) * lambda;

    // Check that this is the correct solution by testing the last element of t.
    if (t.z() < 0)
    {
        // Switch to other solution.
        t = -t;
        R.topLeftCorner<3, 2>() *= -1.0;
    }

    Eigen::Matrix4d cameraPose{Eigen::Matrix4d::Identity()};
    cameraPose.topLeftCorner<3, 3>()  = R;
    cameraPose.topRightCorner<3, 1>() = t;
    std::cout << R << std::endl << t << std::endl << cameraPose << std::endl;

    return cameraPose;
}

// Given a template planar image, locates the bounding box in the source image
void Homography::LocateTemplatePlane(const cv::Mat &planeReferenceImg,
                                     const cv::Mat &inputImage,
                                     const cv::Mat &homographyMatrix)
{
    cv::Mat inputImageCpy;
    inputImage.copyTo(inputImageCpy);
    // Get bbox of template image
    std::vector<cv::Point2f> templateBboxCorners{cv::Point2f(0, 0),
                                                 cv::Point2f(0, planeReferenceImg.rows),
                                                 cv::Point2f(planeReferenceImg.cols, planeReferenceImg.rows),
                                                 cv::Point2f(planeReferenceImg.cols, 0)};
    // Warp the template bbox corners with homography
    std::vector<cv::Point2f> warpedTemplateBboxCorners;
    cv::perspectiveTransform(templateBboxCorners, warpedTemplateBboxCorners, homographyMatrix);

    // Draw the warped bbox on the input image
    std::vector<cv::Point> bboxCornersDraw;
    for (const auto corner : warpedTemplateBboxCorners)
    {
        std::cout << corner << " ";
        bboxCornersDraw.push_back(cv::Point(corner));
    }
    std::cout << std::endl;
    cv::polylines(inputImageCpy, bboxCornersDraw, true, cv::Scalar(255, 0, 100), 5);
    cv::imshow("borders", inputImageCpy);
    // Morph the template image bbox with the homography
}

// This function stitches the right image to the left image through homography
cv::Mat Homography::StitchRightToLeft(const cv::Mat &leftImg, const cv::Mat &rightImg)
{

    FindMatches(leftImg, rightImg);
    auto homographyMatrix{ComputeHomography(HOMOGRAPHY_MODE::IMAGE_TO_IMAGE)};

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

    return stitched;
}

PlaneReference Homography::GetPlaneRefCopy() const
{
    // NOTE: this is not a deep copy operation as there are dynamically allocated data
    return PlaneReference(*planeReference_);
}

cv::Matx33d Homography::GetCameraInstrinsics() const
{
    return this->K_;
}

} // namespace homography