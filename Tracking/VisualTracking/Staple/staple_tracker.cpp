#include "staple_tracker.hpp"
#include "fhog.h"
#include <iomanip>

// mexResize got different results using different OpenCV, it's not trustable
// I found this bug by running vot2015/tunnel, it happened when frameno+1==22 after frameno+1==21
void STAPLE_TRACKER::mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz)
{
    cv::resize(im, output, newsz, 0, 0, cv::INTER_LINEAR);
}

staple_cfg STAPLE_TRACKER::default_parameters_staple(staple_cfg cfg)
{
    return cfg;
}

void STAPLE_TRACKER::initializeAllAreas(const cv::Mat &im, const cv::Size &target_size)
{
    // we want a regular frame surrounding the object
    double avg_dim = (target_size.width + target_size.height) / 2.0;

    bg_area_.width  = round(target_size.width + avg_dim);
    bg_area_.height = round(target_size.height + avg_dim);

    // pick a "safe" region smaller than bbox to avoid mislabeling
    fg_area_.width  = round(target_size.width - avg_dim * cfg.inner_padding);
    fg_area_.height = round(target_size.height - avg_dim * cfg.inner_padding);

    // saturate to image size
    cv::Size imsize = im.size();

    bg_area_.width  = std::min(bg_area_.width, imsize.width - 1);
    bg_area_.height = std::min(bg_area_.height, imsize.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    bg_area_.width  = bg_area_.width - (bg_area_.width - target_size.width) % 2;
    bg_area_.height = bg_area_.height - (bg_area_.height - target_size.height) % 2;

    fg_area_.width  = fg_area_.width + (bg_area_.width - fg_area_.width) % 2;
    fg_area_.height = fg_area_.height + (bg_area_.height - fg_area_.width) % 2;

    // Compute the rectangle with (or close to) params.fixed_area
    // and same aspect ratio as the target bbox
    area_resize_factor_  = sqrt(cfg.fixed_area / double(bg_area_.width * bg_area_.height));
    norm_bg_area_.width  = round(bg_area_.width * area_resize_factor_);
    norm_bg_area_.height = round(bg_area_.height * area_resize_factor_);

    // Correlation Filter (HOG) feature space
    // Make it smaller than the norm_bg_area if HOG cell size is > 1
    cf_response_size_.width  = floor(norm_bg_area_.width / cfg.hog_cell_size);
    cf_response_size_.height = floor(norm_bg_area_.height / cfg.hog_cell_size);

    // given the norm BG area, which is the corresponding target w and h?
    double norm_target_sz_w = 0.75 * norm_bg_area_.width - 0.25 * norm_bg_area_.height;
    double norm_target_sz_h = 0.75 * norm_bg_area_.height - 0.25 * norm_bg_area_.width;

    norm_target_sz_.width  = round(norm_target_sz_w);
    norm_target_sz_.height = round(norm_target_sz_h);

    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width  = floor((norm_bg_area_.width - norm_target_sz_.width) / 2.0);
    norm_pad.height = floor((norm_bg_area_.height - norm_target_sz_.height) / 2.0);

    int radius = floor(fmin(norm_pad.width, norm_pad.height));

    // norm_delta_area_ is the number of rectangles that are considered.
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    norm_delta_area_ = cv::Size((2 * radius + 1), (2 * radius + 1));

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_sz_) has size norm_delta_area_.
    norm_pwp_search_area_.width  = norm_target_sz_.width + norm_delta_area_.width - 1;
    norm_pwp_search_area_.height = norm_target_sz_.height + norm_delta_area_.height - 1;
}

// GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
//   Returns sub-window of image "im" centered at "centerCoor" ([y, x] coordinates),
//   with size "model_sz" ([height, width]). If any pixels are outside of the image,
//   they will replicate the values at the borders
void STAPLE_TRACKER::getSubwindow(const cv::Mat    &im,
                                  cv::Point_<float> centerCoor,
                                  cv::Size          model_sz,
                                  cv::Size          scaled_sz,
                                  cv::Mat          &output)
{
    // make sure the size is not to small
    scaled_sz.width  = fmax(scaled_sz.width, 2);
    scaled_sz.height = fmax(scaled_sz.height, 2);

    cv::Mat subWindow;

    cv::Point lefttop(
        std::min(im.cols - 1, std::max(-scaled_sz.width + 1, int(centerCoor.x + 1 - scaled_sz.width / 2.0 + 0.5))),
        std::min(im.rows - 1, std::max(-scaled_sz.height + 1, int(centerCoor.y + 1 - scaled_sz.height / 2.0 + 0.5))));

    cv::Point rightbottom(std::max(0, int(lefttop.x + scaled_sz.width - 1)),
                          std::max(0, int(lefttop.y + scaled_sz.height - 1)));

    cv::Point lefttopLimit(std::max(lefttop.x, 0), std::max(lefttop.y, 0));
    cv::Point rightbottomLimit(std::min(rightbottom.x, im.cols - 1), std::min(rightbottom.y, im.rows - 1));

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    im(roiRect).copyTo(subWindow);

    int top    = lefttopLimit.y - lefttop.y;
    int bottom = rightbottom.y - rightbottomLimit.y + 1;
    int left   = lefttopLimit.x - lefttop.x;
    int right  = rightbottom.x - rightbottomLimit.x + 1;

    cv::copyMakeBorder(subWindow, subWindow, top, bottom, left, right, cv::BORDER_REPLICATE);

    mexResize(subWindow, output, model_sz);
}

// UPDATEHISTMODEL create new models for foreground and background or update the current ones
void STAPLE_TRACKER::updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp)
{
    // Get BG (frame around target_sz) and FG masks (inner portion of target_sz)

    ////////////////////////////////////////////////////////////////////////
    cv::Size pad_offset1;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset1.width  = (bg_area_.width - target_sz.width) / 2;
    pad_offset1.height = (bg_area_.height - target_sz.height) / 2;

    // difference between bg_area_ and target_sz has to be even
    if (((pad_offset1.width == round(pad_offset1.width)) && (pad_offset1.height != round(pad_offset1.height))) ||
        ((pad_offset1.width != round(pad_offset1.width)) && (pad_offset1.height == round(pad_offset1.height))))
    {
        assert(0);
    }

    pad_offset1.width  = fmax(pad_offset1.width, 1);
    pad_offset1.height = fmax(pad_offset1.height, 1);

    cv::Mat bg_mask(bg_area_, CV_8UC1, cv::Scalar(1)); // init bg_mask

    cv::Rect pad1_rect(pad_offset1.width,
                       pad_offset1.height,
                       bg_area_.width - 2 * pad_offset1.width,
                       bg_area_.height - 2 * pad_offset1.height);

    bg_mask(pad1_rect) = false;

    cv::Size pad_offset2;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset2.width  = (bg_area_.width - fg_area_.width) / 2;
    pad_offset2.height = (bg_area_.height - fg_area_.height) / 2;

    // difference between bg_area_ and fg_area_ has to be even
    if (((pad_offset2.width == round(pad_offset2.width)) && (pad_offset2.height != round(pad_offset2.height))) ||
        ((pad_offset2.width != round(pad_offset2.width)) && (pad_offset2.height == round(pad_offset2.height))))
    {
        assert(0);
    }

    pad_offset2.width  = fmax(pad_offset2.width, 1);
    pad_offset2.height = fmax(pad_offset2.height, 1);

    cv::Mat fg_mask(bg_area_, CV_8UC1, cv::Scalar(0)); // init fg_mask

    cv::Rect pad2_rect(pad_offset2.width,
                       pad_offset2.height,
                       bg_area_.width - 2 * pad_offset2.width,
                       bg_area_.height - 2 * pad_offset2.height);

    fg_mask(pad2_rect) = true;

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;

    mexResize(fg_mask, fg_mask_new, norm_bg_area_);
    mexResize(bg_mask, bg_mask_new, norm_bg_area_);

    int          imgCount   = 1;
    int          dims       = 3;
    const int    sizes[]    = {cfg.n_bins, cfg.n_bins, cfg.n_bins};
    const int    channels[] = {0, 1, 2};
    float        bRange[]   = {0, 256};
    float        gRange[]   = {0, 256};
    float        rRange[]   = {0, 256};
    const float *ranges[]   = {bRange, gRange, rRange};

    if (cfg.grayscale_sequence)
    {
        dims = 1;
    }

    // (TRAIN) BUILD THE MODEL
    if (new_model)
    {
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        bg_hist = bg_hist / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
        fg_hist = fg_hist / fgtotal;
    }
    else
    { // update the model
        cv::MatND bg_hist_tmp;
        cv::MatND fg_hist_tmp;

        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        bg_hist_tmp = bg_hist_tmp / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
        fg_hist_tmp = fg_hist_tmp / fgtotal;

        // xxx
        bg_hist = (1 - learning_rate_pwp) * bg_hist + learning_rate_pwp * bg_hist_tmp;
        fg_hist = (1 - learning_rate_pwp) * fg_hist + learning_rate_pwp * fg_hist_tmp;
    }
}

void STAPLE_TRACKER::CalculateHann(cv::Size sz, cv::Mat &output)
{
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);

    float *p1 = temp1.ptr<float>(0);
    float *p2 = temp2.ptr<float>(0);

    for (int i = 0; i < sz.width; ++i)
        p1[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.width - 1)));

    for (int i = 0; i < sz.height; ++i)
        p2[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.height - 1)));

    output = temp2.t() * temp1;
}

void meshgrid(const cv::Range xr, const cv::Range yr, cv::Mat &outX, cv::Mat &outY)
{
    std::vector<int> x, y;

    for (int i = xr.start; i <= xr.end; i++)
        x.push_back(i);
    for (int i = yr.start; i <= yr.end; i++)
        y.push_back(i);

    repeat(cv::Mat(x).t(), y.size(), 1, outX);
    repeat(cv::Mat(y), 1, x.size(), outY);
}

// GAUSSIANRESPONSE create the (fixed) target response of the correlation filter response
void STAPLE_TRACKER::gaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output)
{
    // half = floor((rect_size-1) / 2);
    // i_range = -half(1):half(1);
    // j_range = -half(2):half(2);
    // [i, j] = ndgrid(i_range, j_range);
    cv::Size half;

    half.width  = floor((rect_size.width - 1) / 2);
    half.height = floor((rect_size.height - 1) / 2);

    cv::Range i_range(-half.width, rect_size.width - (1 + half.width));
    cv::Range j_range(-half.height, rect_size.height - (1 + half.height));
    cv::Mat   i, j;

    meshgrid(i_range, j_range, i, j);

    // i_mod_range = mod_one(i_range, rect_size(1));
    // j_mod_range = mod_one(j_range, rect_size(2));

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    for (int k = i_range.start; k <= i_range.end; k++)
    {
        int val = (int)(k - 1 + rect_size.width) % (int)rect_size.width;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++)
    {
        int val = (int)(k - 1 + rect_size.height) % (int)rect_size.height;
        j_mod_range.push_back(val);
    }

    // y = zeros(rect_size);
    // y(i_mod_range, j_mod_range) = exp(-(i.^2 + j.^2) / (2 * sigma^2));

    output = cv::Mat(rect_size.height, rect_size.width, CV_32FC2);

    for (int jj = 0; jj < rect_size.height; jj++)
    {
        int j_idx = j_mod_range[jj];
        assert(j_idx < rect_size.height);

        for (int ii = 0; ii < rect_size.width; ii++)
        {
            int i_idx = i_mod_range[ii];
            assert(i_idx < rect_size.width);

            cv::Vec2f val(exp(-(i.at<int>(jj, ii) * i.at<int>(jj, ii) + j.at<int>(jj, ii) * j.at<int>(jj, ii)) /
                              (2 * sigma * sigma)),
                          0);
            output.at<cv::Vec2f>(j_idx, i_idx) = val;
        }
    }
}

void STAPLE_TRACKER::tracker_staple_initialize(const cv::Mat &im, cv::Rect_<float> bbox)
{
    int n = im.channels();

    if (n == 1)
    {
        cfg.grayscale_sequence = true;
    }

    // xxx: only support 3 channels, TODO: fix updateHistModel
    //assert(!cfg.grayscale_sequence);

    double bbox_w = bbox.width;
    double bbox_h = bbox.height;

    cv::Point_<float> init_bbox_center;
    init_bbox_center.x = bbox.x + bbox.width / 2.0;
    init_bbox_center.y = bbox.y + bbox.height / 2.0;

    cv::Size target_size;
    target_size.width  = round(bbox_w);
    target_size.height = round(bbox_h);
    std::cout << "target_size: " << target_size.width << " " << target_size.height << std::endl;

    initializeAllAreas(im, target_size);

    bbox_center = init_bbox_center;
    target_sz   = target_size;

    // patch of the target + padding
    cv::Mat patch_padded;

    getSubwindow(im, bbox_center, norm_bg_area_, bg_area_, patch_padded);
    cv::imshow("patch_padded", patch_padded);
    cv::waitKey(0);

    // initialize hist model
    updateHistModel(true, patch_padded);

    CalculateHann(cf_response_size_, hann_window);

    // gaussian-shaped desired response, centred in (1,1)
    // bandwidth proportional to target size
    double output_sigma =
        sqrt(norm_target_sz_.width * norm_target_sz_.height) * cfg.output_sigma_factor / cfg.hog_cell_size;

    cv::Mat y;
    gaussianResponse(cf_response_size_, output_sigma, y);
    cv::dft(y, yf);

    // SCALE ADAPTATION INITIALIZATION
    if (cfg.scale_adaptation)
    {
        // Code from DSST
        scale_factor      = 1;
        base_target_sz    = target_sz; // xxx
        float scale_sigma = sqrt(cfg.num_scales) * cfg.scale_sigma_factor;

        cv::Mat ys = cv::Mat(1, cfg.num_scales, CV_32FC2);
        for (int i = 0; i < cfg.num_scales; i++)
        {
            cv::Vec2f val((i + 1) - ceil(cfg.num_scales / 2.0f), 0.f);
            val[0]              = exp(-0.5 * (val[0] * val[0]) / (scale_sigma * scale_sigma));
            ys.at<cv::Vec2f>(i) = val;

            // SS = (1:p.num_scales) - ceil(p.num_scales/2);
            // ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        }

        cv::dft(ys, ysf, cv::DFT_ROWS);
        //std::cout << ysf << std::endl;

        scale_window = cv::Mat(1, cfg.num_scales, CV_32FC1);
        if (cfg.num_scales % 2 == 0)
        {
            for (int i = 0; i < cfg.num_scales + 1; ++i)
            {
                if (i > 0)
                {
                    scale_window.at<float>(i - 1) = 0.5 * (1 - cos(CV_2PI * i / (cfg.num_scales + 1 - 1)));
                }
            }
        }
        else
        {
            for (int i = 0; i < cfg.num_scales; ++i)
            {
                scale_window.at<float>(i) = 0.5 * (1 - cos(CV_2PI * i / (cfg.num_scales - 1)));
            }
        }

        scale_factors = cv::Mat(1, cfg.num_scales, CV_32FC1);
        for (int i = 0; i < cfg.num_scales; i++)
        {
            scale_factors.at<float>(i) = pow(cfg.scale_step, (ceil(cfg.num_scales / 2.0) - (i + 1)));
        }

        //std::cout << scale_factors << std::endl;

        //ss = 1:p.num_scales;
        //scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

        if ((cfg.scale_model_factor * cfg.scale_model_factor) * (norm_target_sz_.width * norm_target_sz_.height) >
            cfg.scale_model_max_area)
        {
            cfg.scale_model_factor = sqrt(cfg.scale_model_max_area / (norm_target_sz_.width * norm_target_sz_.height));
        }

        //std::cout << cfg.scale_model_factor << std::endl;

        scale_model_sz.width  = floor(norm_target_sz_.width * cfg.scale_model_factor);
        scale_model_sz.height = floor(norm_target_sz_.height * cfg.scale_model_factor);

        //std::cout << scale_model_sz << std::endl;

        // find maximum and minimum scales
        min_scale_factor =
            pow(cfg.scale_step, ceil(log(std::max(5.0 / bg_area_.width, 5.0 / bg_area_.height)) / log(cfg.scale_step)));
        max_scale_factor =
            pow(cfg.scale_step,
                floor(log(std::min(im.cols / (float)target_sz.width, im.rows / (float)target_sz.height)) /
                      log(cfg.scale_step)));
        //min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area_)) / log(p.scale_step));
        //max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

        //std::cout << min_scale_factor << " " << max_scale_factor << std::endl;
    }
}

// code from DSST
void STAPLE_TRACKER::getFeatureMap(cv::Mat &im_patch, const char *feature_type, cv::MatND &output)
{
    assert(!strcmp(feature_type, "fhog"));

    fhog28(output, im_patch, cfg.hog_cell_size, 9);

    int w = cf_response_size_.width;
    int h = cf_response_size_.height;

    // hog28 already generate this matrix of (w,h,28)
    // out = zeros(h, w, 28, 'single');
    // out(:,:,2:28) = temp(:,:,1:27);

    cv::Mat new_im_patch;

    if (cfg.hog_cell_size > 1)
    {
        cv::Size newsz(w, h);

        mexResize(im_patch, new_im_patch, newsz);
    }
    else
    {
        new_im_patch = im_patch;
    }

    cv::Mat grayimg;

    if (new_im_patch.channels() > 1)
    {
        cv::cvtColor(new_im_patch, grayimg, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayimg = new_im_patch;
    }

    // out(:,:,1) = single(im_patch)/255 - 0.5;

    float alpha = 1. / 255.0;
    float betta = 0.5;

    typedef cv::Vec<float, 28> Vecf28;

    for (int j = 0; j < h; ++j)
    {
        Vecf28      *pDst  = output.ptr<Vecf28>(j);
        const float *pHann = hann_window.ptr<float>(j);
        const uchar *pGray = grayimg.ptr<uchar>(j);

        for (int i = 0; i < w; ++i)
        {
            // apply Hann window
            Vecf28 &val = pDst[0];

            val    = val * pHann[0];
            val[0] = (alpha * pGray[0] - betta) * pHann[0];

            ++pDst;
            ++pHann;
            ++pGray;
        }
    }
}

void matsplit(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit)
{
    int w  = xt.cols;
    int h  = xt.rows;
    int cn = xt.channels();

    assert(cn == 28);

    for (int k = 0; k < cn; k++)
    {
        cv::Mat dim = cv::Mat(h, w, CV_32FC2);

        for (int j = 0; j < h; ++j)
        {
            float       *pDst = dim.ptr<float>(j);
            const float *pSrc = xt.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                pDst[0] = pSrc[k];
                pDst[1] = 0.0f;

                pSrc += cn;
                pDst += 2;
            }
        }

        xtsplit.push_back(dim);
    }
}

// GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
//   Returns sub-window of image "im" centered at "centerCoor" ([y, x] coordinates),
//   with size "model_sz" ([height, width]). If any pixels are outside of the image,
//   they will replicate the values at the borders
void STAPLE_TRACKER::getSubwindowFloor(const cv::Mat    &im,
                                       cv::Point_<float> centerCoor,
                                       cv::Size          model_sz,
                                       cv::Size          scaled_sz,
                                       cv::Mat          &output)
{
    // make sure the size is not to small
    scaled_sz.width  = fmax(scaled_sz.width, 2);
    scaled_sz.height = fmax(scaled_sz.height, 2);

    cv::Mat subWindow;

    cv::Point lefttop(
        std::min(im.cols - 1, std::max(-scaled_sz.width + 1, int(centerCoor.x + 1) - int(scaled_sz.width / 2.0))),
        std::min(im.rows - 1, std::max(-scaled_sz.height + 1, int(centerCoor.y + 1) - int(scaled_sz.height / 2.0))));

    cv::Point rightbottom(std::max(0, int(lefttop.x + scaled_sz.width - 1)),
                          std::max(0, int(lefttop.y + scaled_sz.height - 1)));

    cv::Point lefttopLimit(std::max(lefttop.x, 0), std::max(lefttop.y, 0));
    cv::Point rightbottomLimit(std::min(rightbottom.x, im.cols - 1), std::min(rightbottom.y, im.rows - 1));

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    im(roiRect).copyTo(subWindow);

    // imresize(subWindow, output, model_sz, 'bilinear', 'AntiAliasing', false)
    mexResize(subWindow, output, model_sz);
}

// code from DSST
void STAPLE_TRACKER::getScaleSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Mat &output)
{
    int ch    = 0;
    int total = 0;

    for (int s = 0; s < cfg.num_scales; s++)
    {
        cv::Size_<float> patch_sz;

        patch_sz.width  = floor(base_target_sz.width * scale_factor * scale_factors.at<float>(s));
        patch_sz.height = floor(base_target_sz.height * scale_factor * scale_factors.at<float>(s));

        cv::Mat im_patch_resized;
        getSubwindowFloor(im, centerCoor, scale_model_sz, patch_sz, im_patch_resized);

        // extract scale features
        cv::MatND temp;
        fhog31(temp, im_patch_resized, cfg.hog_cell_size, 9);

        if (s == 0)
        {
            ch    = temp.channels();
            total = temp.cols * temp.rows * ch;

            output = cv::Mat(total, cfg.num_scales, CV_32FC2);
        }

        int tempw  = temp.cols;
        int temph  = temp.rows;
        int tempch = temp.channels();

        int count = 0;

        float scaleWnd = scale_window.at<float>(s);

        float *outData = (float *)output.data;

        // window
        for (int j = 0; j < temph; ++j)
        {
            const float *tmpData = temp.ptr<float>(j);

            for (int i = 0; i < tempw; ++i)
            {
                for (int k = 0; k < tempch; ++k)
                {
                    outData[(count * cfg.num_scales + s) * 2 + 0] = tmpData[k] * scaleWnd;
                    outData[(count * cfg.num_scales + s) * 2 + 1] = 0.0;

                    count++;
                }
                tmpData += ch;
            }
        }
    }
}

// TRAINING
void STAPLE_TRACKER::tracker_staple_train(const cv::Mat &im, bool first)
{
    // extract patch of size bg_area_ and resize to norm_bg_area_
    cv::Mat im_patch_bg;
    getSubwindow(im, bbox_center, norm_bg_area_, bg_area_, im_patch_bg);

    // compute feature map, of cf_response_size_
    cv::MatND xt;
    getFeatureMap(im_patch_bg, cfg.feature_type, xt);

    // apply Hann window in getFeatureMap
    // xt = bsxfun(@times, hann_window, xt);

    // compute FFT
    // cv::MatND xtf;
    std::vector<cv::Mat> xtsplit;
    std::vector<cv::Mat> xtf; // xtf is splits of xtf

    matsplit(xt, xtsplit);

    for (int i = 0; i < xt.channels(); i++)
    {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

    // FILTER UPDATE
    // Compute expectations over circular shifts,
    // therefore divide by number of pixels.
    // new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size_);
    // new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size_);

    {
        std::vector<cv::Mat> new_hf_num;
        std::vector<cv::Mat> new_hf_den;

        int   w       = xt.cols;
        int   h       = xt.rows;
        float invArea = 1.f / (cf_response_size_.width * cf_response_size_.height);

        for (int ch = 0; ch < xt.channels(); ch++)
        {
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);

            for (int j = 0; j < h; ++j)
            {
                const float *pXTF = xtf[ch].ptr<float>(j);
                const float *pYF  = yf.ptr<float>(j);
                cv::Vec2f   *pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    cv::Vec2f val(pYF[1] * pXTF[1] + pYF[0] * pXTF[0], pYF[0] * pXTF[1] - pYF[1] * pXTF[0]);
                    *pDst = invArea * val;

                    pXTF += 2;
                    pYF += 2;
                    ++pDst;
                }
            }

            new_hf_num.push_back(dim);
        }

        for (int ch = 0; ch < xt.channels(); ch++)
        {
            cv::Mat dim = cv::Mat(h, w, CV_32FC1);

            for (int j = 0; j < h; ++j)
            {
                const float *pXTF = xtf[ch].ptr<float>(j);
                float       *pDst = dim.ptr<float>(j);

                for (int i = 0; i < w; ++i)
                {
                    *pDst = invArea * (pXTF[0] * pXTF[0] + pXTF[1] * pXTF[1]);

                    pXTF += 2;
                    ++pDst;
                }
            }

            new_hf_den.push_back(dim);
        }

        if (first)
        {
            // first frame, train with a single image
            hf_den.assign(new_hf_den.begin(), new_hf_den.end());
            hf_num.assign(new_hf_num.begin(), new_hf_num.end());
        }
        else
        {
            // subsequent frames, update the model by linear interpolation
            for (int ch = 0; ch < xt.channels(); ch++)
            {
                hf_den[ch] = (1 - cfg.learning_rate_cf) * hf_den[ch] + cfg.learning_rate_cf * new_hf_den[ch];
                hf_num[ch] = (1 - cfg.learning_rate_cf) * hf_num[ch] + cfg.learning_rate_cf * new_hf_num[ch];
            }

            updateHistModel(false, im_patch_bg, cfg.learning_rate_pwp);
        }
    }

    // SCALE UPDATE
    if (cfg.scale_adaptation)
    {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, bbox_center, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        cv::Mat new_sf_num;
        cv::Mat new_sf_den;

        int w = xsf.cols;
        int h = xsf.rows;

        new_sf_num = cv::Mat(h, w, CV_32FC2);

        for (int j = 0; j < h; ++j) // xxx
        {
            float *pDst = new_sf_num.ptr<float>(j);

            const float *pXSF = xsf.ptr<float>(j);
            const float *pYSF = ysf.ptr<float>(0);

            for (int i = 0; i < w; ++i)
            {
                pDst[0] = (pYSF[1] * pXSF[1] + pYSF[0] * pXSF[0]);
                pDst[1] = (pYSF[1] * pXSF[0] - pYSF[0] * pXSF[1]);

                pXSF += 2;
                pYSF += 2;
                pDst += 2;
            }
        }

        new_sf_den  = cv::Mat(1, w, CV_32FC1, cv::Scalar(0, 0, 0));
        float *pDst = new_sf_den.ptr<float>(0);

        for (int j = 0; j < h; ++j)
        {
            const float *pSrc = xsf.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                pDst[i] += (pSrc[0] * pSrc[0] + pSrc[1] * pSrc[1]);
                pSrc += 2;
            }
        }

        if (first)
        {
            // first frame, train with a single image
            new_sf_den.copyTo(sf_den);
            new_sf_num.copyTo(sf_num);
        }
        else
        {
            sf_den = (1 - cfg.learning_rate_scale) * sf_den + cfg.learning_rate_scale * new_sf_den;
            sf_num = (1 - cfg.learning_rate_scale) * sf_num + cfg.learning_rate_scale * new_sf_num;
        }
    }

    // update bbox position
    if (first)
    {
        rect_position.x      = bbox_center.x - target_sz.width / 2;
        rect_position.y      = bbox_center.y - target_sz.height / 2;
        rect_position.width  = target_sz.width;
        rect_position.height = target_sz.height;
    }

    frameno += 1;
}

// xxx: improve later
cv::Mat ensure_real(const cv::Mat &complex)
{
    int w = complex.cols;
    int h = complex.rows;

    cv::Mat real = cv::Mat(h, w, CV_32FC1);

    for (int j = 0; j < h; ++j)
    {
        float       *pDst = real.ptr<float>(j);
        const float *pSrc = complex.ptr<float>(j);

        for (int i = 0; i < w; ++i)
        {
            *pDst = *pSrc;
            ++pDst;
            pSrc += 2;
        }
    }

    return real;
}

void STAPLE_TRACKER::cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat &output)
{
    int w = response_cf.cols;
    int h = response_cf.rows;

    // newh and neww must be odd, as we want an exact center
    assert(((response_size.width % 2) == 1) && ((response_size.height % 2) == 1));

    int half_width  = response_size.width / 2;
    int half_height = response_size.height / 2;

    cv::Range i_range(-half_width, response_size.width - (1 + half_width));
    cv::Range j_range(-half_height, response_size.height - (1 + half_height));

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    for (int k = i_range.start; k <= i_range.end; k++)
    {
        int val = (k - 1 + w) % w;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++)
    {
        int val = (k - 1 + h) % h;
        j_mod_range.push_back(val);
    }

    cv::Mat tmp = cv::Mat(response_size.height, response_size.width, CV_32FC1, cv::Scalar(0, 0, 0));

    for (int j = 0; j < response_size.height; j++)
    {
        int j_idx = j_mod_range[j];
        assert(j_idx < h);

        float       *pDst = tmp.ptr<float>(j);
        const float *pSrc = response_cf.ptr<float>(j_idx);

        for (int i = 0; i < response_size.width; i++)
        {
            int i_idx = i_mod_range[i];
            assert(i_idx < w);

            *pDst = pSrc[i_idx];
            ++pDst;
        }
    }
    output = tmp;
}

// GETCOLOURMAP computes pixel-wise probabilities (PwP) given PATCH and models BG_HIST and FG_HIST
void STAPLE_TRACKER::getColourMap(const cv::Mat &patch, cv::Mat &output)
{
    // check whether the patch has 3 channels
    int h = patch.rows;
    int w = patch.cols;
    int d = patch.channels();

    // figure out which bin each pixel falls into
    int bin_width = 256 / cfg.n_bins;

    // convert image to d channels array
    //patch_array = reshape(double(patch), w*h, d);

    output = cv::Mat(h, w, CV_32FC1);

    if (!cfg.grayscale_sequence)
    {
        for (int j = 0; j < h; ++j)
        {
            const uchar *pSrc = patch.ptr<uchar>(j);
            float       *pDst = output.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                int b1 = pSrc[0] / bin_width;
                int b2 = pSrc[1] / bin_width;
                int b3 = pSrc[2] / bin_width;

                float *histd = (float *)bg_hist.data;
                float  probg = histd[b1 * cfg.n_bins * cfg.n_bins + b2 * cfg.n_bins + b3];

                histd       = (float *)fg_hist.data;
                float profg = histd[b1 * cfg.n_bins * cfg.n_bins + b2 * cfg.n_bins + b3];

                // xxx
                *pDst = profg / (profg + probg);

                isnan(*pDst) && (*pDst = 0.0);

                pSrc += d;
                ++pDst;

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }
        }
    }
    else
    {
        for (int j = 0; j < h; j++)
        {
            const uchar *pSrc = patch.ptr<uchar>(j);
            float       *pDst = output.ptr<float>(j);

            for (int i = 0; i < w; i++)
            {
                int b = *pSrc;

                float *histd = (float *)bg_hist.data;
                float  probg = histd[b];

                histd       = (float *)fg_hist.data;
                float profg = histd[b];

                // xxx
                *pDst = profg / (profg + probg);

                isnan(*pDst) && (*pDst = 0.0);

                pSrc += d;
                ++pDst;
            }
        }
    }
}

// GETCENTERLIKELIHOOD computes the sum over rectangles of size M.
void STAPLE_TRACKER::getCenterLikelihood(const cv::Mat &object_likelihood,
                                         cv::Size       normalize_target_size,
                                         cv::Mat       &center_likelihood)
{
    // CENTER_LIKELIHOOD is the 'colour response'
    int   h       = object_likelihood.rows;
    int   w       = object_likelihood.cols;
    int   n1      = w - normalize_target_size.width + 1;
    int   n2      = h - normalize_target_size.height + 1;
    float invArea = 1.f / (normalize_target_size.width * normalize_target_size.height);

    // integral images
    cv::Mat integral;
    cv::integral(object_likelihood, integral);

    cv::Mat integral_viz;
    cv::normalize(integral, integral_viz, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("integral", integral_viz);

    center_likelihood = cv::Mat(n2, n1, CV_32FC1);

    for (int j = 0; j < n2; ++j)
    {
        float *pLike = reinterpret_cast<float *>(center_likelihood.ptr(j));

        for (int i = 0; i < n1; ++i)
        {
            *pLike = invArea * (integral.at<double>(j, i) +
                                integral.at<double>(j + normalize_target_size.height, i + normalize_target_size.width) -
                                integral.at<double>(j, i + normalize_target_size.width) -
                                integral.at<double>(j + normalize_target_size.height, i));
            ++pLike;
        }
    }

    cv::Mat center_likelihood_viz;
    cv::normalize(center_likelihood, center_likelihood_viz, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("Likelihood", center_likelihood_viz);
}

void STAPLE_TRACKER::mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response)
{
    double alpha = cfg.merge_factor;
    //const char *merge_method = cfg.merge_method;

    // MERGERESPONSES interpolates the two responses with the hyperparameter ALPHA
    response = (1 - alpha) * response_cf + alpha * response_pwp;

    // response = (1 - alpha) * response_cf + alpha * response_pwp;
}

// TESTING step
cv::Rect STAPLE_TRACKER::tracker_staple_update(const cv::Mat &im)
{
    // extract patch of size bg_area_ and resize to norm_bg_area_
    cv::Mat im_patch_cf;
    getSubwindow(im, bbox_center, norm_bg_area_, bg_area_, im_patch_cf);

    cv::Size pwp_search_area;

    pwp_search_area.width  = round(norm_pwp_search_area_.width / area_resize_factor_);
    pwp_search_area.height = round(norm_pwp_search_area_.height / area_resize_factor_);

    // extract patch of size pwp_search_area and resize to norm_pwp_search_area_
    getSubwindow(im, bbox_center, norm_pwp_search_area_, pwp_search_area, im_patch_pwp);

    // compute feature map
    cv::MatND xt_windowed;
    getFeatureMap(im_patch_cf, cfg.feature_type, xt_windowed);

    // apply Hann window in getFeatureMap

    // compute FFT
    // cv::MatND xtf;
    std::vector<cv::Mat> xtsplit;
    std::vector<cv::Mat> xtf; // xtf is splits of xtf

    matsplit(xt_windowed, xtsplit);

    for (int i = 0; i < xt_windowed.channels(); i++)
    {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

    std::vector<cv::Mat> hf;
    const int            w = xt_windowed.cols;
    const int            h = xt_windowed.rows;

    // Correlation between filter and test patch gives the response
    // Solve diagonal system per pixel.
    if (cfg.den_per_channel)
    {
        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);

            for (int j = 0; j < h; ++j)
            {
                const cv::Vec2f *pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                const float     *pDen = hf_den[ch].ptr<float>(j);
                cv::Vec2f       *pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    pDst[i] = pSrc[i] / (pDen[i] + cfg.lambda);
                }
            }

            hf.push_back(dim);
        }
    }
    else
    {
        std::vector<float> DIM1(w * h, cfg.lambda);

        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            float *pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j)
            {
                const float *pDen = hf_den[ch].ptr<float>(j);
                for (int i = 0; i < w; ++i)
                {
                    *pDim1 += pDen[i];
                    ++pDim1;
                }
            }
        }

        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            cv::Mat      dim   = cv::Mat(h, w, CV_32FC2);
            const float *pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j)
            {
                const cv::Vec2f *pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                cv::Vec2f       *pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    *pDst = *pSrc / *pDim1;
                    ++pDim1;
                    ++pDst;
                    ++pSrc;
                }
            }

            hf.push_back(dim);
        }
    }

    cv::Mat response_cff = cv::Mat(h, w, CV_32FC2);

    for (int j = 0; j < h; j++)
    {
        cv::Vec2f *pDst = response_cff.ptr<cv::Vec2f>(j);

        for (int i = 0; i < w; i++)
        {
            float sum  = 0.0;
            float sumi = 0.0;

            for (size_t ch = 0; ch < hf.size(); ch++)
            {
                cv::Vec2f pHF  = hf[ch].at<cv::Vec2f>(j, i);
                cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(j, i);

                sum += (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                sumi += (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);

                // assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
            }

            *pDst = cv::Vec2f(sum, sumi);
            ++pDst;
        }
    }

    cv::Mat response_cfi;
    cv::dft(response_cff, response_cfi, cv::DFT_SCALE | cv::DFT_INVERSE);
    cv::Mat response_cf = ensure_real(response_cfi);

    // response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));

    // Crop square search region (in feature pixels).
    cv::Size newsz = norm_delta_area_;
    newsz.width    = floor(newsz.width / cfg.hog_cell_size);
    newsz.height   = floor(newsz.height / cfg.hog_cell_size);

    (newsz.width % 2 == 0) && (newsz.width -= 1);
    (newsz.height % 2 == 0) && (newsz.height -= 1);

    cropFilterResponse(response_cf, newsz, response_cf);

    if (cfg.hog_cell_size > 1)
    {
        cv::Mat temp;
        mexResize(response_cf, temp, norm_delta_area_);
        response_cf = temp; // xxx: low performance
    }

    cv::Mat likelihood_map;
    getColourMap(im_patch_pwp, likelihood_map);
    //[likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);

    // each pixel of response_pwp loosely represents the likelihood that
    // the target (of size norm_target_sz_) is centred on it
    cv::Mat response_pwp;
    getCenterLikelihood(likelihood_map, norm_target_sz_, response_pwp);

    // ESTIMATION
    cv::Mat response;
    mergeResponses(response_cf, response_pwp, response);

    double    maxVal = 0;
    cv::Point maxLoc;

    cv::minMaxLoc(response, nullptr, &maxVal, nullptr, &maxLoc);

    float centerx = (1 + norm_delta_area_.width) / 2 - 1;
    float centery = (1 + norm_delta_area_.height) / 2 - 1;

    bbox_center.x += (maxLoc.x - centerx) / area_resize_factor_;
    bbox_center.y += (maxLoc.y - centery) / area_resize_factor_;

    // Report current location
    cv::Rect_<float> location;

    location.x      = bbox_center.x - target_sz.width / 2.0;
    location.y      = bbox_center.y - target_sz.height / 2.0;
    location.width  = target_sz.width;
    location.height = target_sz.height;

    // SCALE SPACE SEARCH
    if (cfg.scale_adaptation)
    {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, bbox_center, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        const int w = xsf.cols;
        const int h = xsf.rows;

        cv::Mat scale_responsef = cv::Mat(1, w, CV_32FC2, cv::Scalar(0, 0, 0));

        for (int j = 0; j < h; ++j)
        {
            const float *pXSF    = xsf.ptr<float>(j);
            const float *pXSFNUM = sf_num.ptr<float>(j);
            const float *pDen    = sf_den.ptr<float>(0);
            float       *pscale  = scale_responsef.ptr<float>(0);

            for (int i = 0; i < w; ++i)
            {
                float invDen = 1.f / (*pDen + cfg.lambda);

                pscale[0] += invDen * (pXSFNUM[0] * pXSF[0] - pXSFNUM[1] * pXSF[1]);
                pscale[1] += invDen * (pXSFNUM[0] * pXSF[1] + pXSFNUM[1] * pXSF[0]);

                pscale += 2;
                pXSF += 2;
                pXSFNUM += 2;
                ++pDen;
            }
        }

        cv::Mat scale_response;
        cv::dft(scale_responsef, scale_response, cv::DFT_SCALE | cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

        //scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));

        double    maxVal = 0;
        cv::Point maxLoc;

        cv::minMaxLoc(scale_response, nullptr, &maxVal, nullptr, &maxLoc);

        int recovered_scale = maxLoc.x;

        // set the scale
        scale_factor = scale_factor * scale_factors.at<float>(recovered_scale);

        if (scale_factor < min_scale_factor)
        {
            scale_factor = min_scale_factor;
        }
        else if (scale_factor > max_scale_factor)
        {
            scale_factor = max_scale_factor;
        }

        // use new scale to update bboxes for target, filter, bg and fg models
        target_sz.width  = round(base_target_sz.width * scale_factor);
        target_sz.height = round(base_target_sz.height * scale_factor);

        float avg_dim = (target_sz.width + target_sz.height) / 2.0;

        bg_area_.width  = round(target_sz.width + avg_dim);
        bg_area_.height = round(target_sz.height + avg_dim);

        (bg_area_.width > im.cols) && (bg_area_.width = im.cols - 1);
        (bg_area_.height > im.rows) && (bg_area_.height = im.rows - 1);

        bg_area_.width  = bg_area_.width - (bg_area_.width - target_sz.width) % 2;
        bg_area_.height = bg_area_.height - (bg_area_.height - target_sz.height) % 2;

        fg_area_.width  = round(target_sz.width - avg_dim * cfg.inner_padding);
        fg_area_.height = round(target_sz.height - avg_dim * cfg.inner_padding);

        fg_area_.width  = fg_area_.width + int(bg_area_.width - fg_area_.width) % 2;
        fg_area_.height = fg_area_.height + int(bg_area_.height - fg_area_.height) % 2;

        // Compute the rectangle with (or close to) params.fixed_area and
        // same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor_ = sqrt(cfg.fixed_area / (float)(bg_area_.width * bg_area_.height));
    }

    return location;
}
