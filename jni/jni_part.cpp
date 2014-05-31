#include <jni.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv* env, jobject thisObj, jobject processAddr, jlong errorCodeAddr);

static void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


// Default command line args
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "orb";
string ba_cost_func = "ray";
string ba_refine_mask = "_____";//"xxxxx";
bool do_wave_correct = false;//true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "plane";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "no";// "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv* env, jobject thisObj, jobject processAddr, jlong errorCodeAddr)
{
	jclass ArrayList_class = env->FindClass("java/util/ArrayList");
	jclass Long_class = env->FindClass("java/lang/Long");
	jmethodID Size_method = env->GetMethodID(ArrayList_class, "size", "()I" );
	jmethodID Get_method = env->GetMethodID(ArrayList_class, "get", "(I)Ljava/lang/Object;" );
	jmethodID Long_method = env->GetMethodID(Long_class, "longValue", "()J" );
	Mat& errorCode = *(Mat*)errorCodeAddr;
	int NumElts = env->CallIntMethod(processAddr, Size_method);
	long addr_i;
	vector<Mat> MatImages;
	vector<int> img_names;
	vector<ImageFeatures> features;
	vector<Mat> images;
	vector<Size> full_img_sizes;
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	double seam_work_aspect = 1;

	for (int i = 0; i < NumElts; i++) {
		addr_i = env->CallLongMethod(env->CallObjectMethod(processAddr, Get_method, i), Long_method);
		MatImages.push_back(*(Mat*)addr_i);
		img_names.push_back(i);
	}

    cv::setBreakOnError(true);

    Ptr<FeaturesFinder> finder;
    finder = new OrbFeaturesFinder();
    errorCode.at<double>(0, 0) = 0;
    Mat full_img, img;
    for (int i = 0; i < NumElts; ++i)
    {
    	full_img = MatImages[i];
    	full_img_sizes.push_back(full_img.size());

        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        features.push_back(ImageFeatures());
        (*finder)(img, features[i]);
        ImageFeatures& featuresi = features[i];
        featuresi.img_idx = i;
        resize(full_img, img, Size(), seam_scale, seam_scale);
        images.push_back(img.clone());
    }

    finder->collectGarbage();
    full_img.release();
    img.release();
    errorCode.at<double>(0, 0) = 10;


    ImageFeatures& featuresi = features[0];

    //test with size 7
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<int> img_names_subset;
    vector<Size> full_img_sizes_subset;


    // Leave only images we are sure are from the same panorama
    for (size_t i = 0; i < indices.size(); ++i)
    {
    	int indicesi = indices[i];
        img_names_subset.push_back(indicesi);
        Mat& imagesi = images[indices[i]];
    	img_subset.push_back(imagesi);
    	Size& sizei = full_img_sizes[indices[i]];
        full_img_sizes_subset.push_back(sizei);
    }
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    NumElts = static_cast<int>(img_names.size());
    if (NumElts < 2) {
    	return;
    }
    errorCode.at<double>(0, 0) = 20;

    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        CameraParams& camerasi = cameras[i];
        camerasi.R.convertTo(R, CV_32F);
        camerasi.R = R;
    }
    errorCode.at<double>(0, 0) = 30;

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();

    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        CameraParams& camerasi = cameras[i];
        focals.push_back(camerasi.focal);
    }
    errorCode.at<double>(0, 0) = 70;

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i) {
        	CameraParams& camerasi = cameras[i];
            rmats.push_back(camerasi.R);
        }
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i) {
        	CameraParams& camerasi = cameras[i];
            camerasi.R = rmats[i];
        }
    }
    errorCode.at<double>(0, 0) = 75;

    vector<Point> corners(NumElts);
    vector<Mat> masks_warped(NumElts);
    vector<Mat> images_warped(NumElts);
    vector<Size> sizes(NumElts);
    vector<Mat> masks(NumElts);

    // Preapre images masks
    for (int i = 0; i < NumElts; ++i)
    {
    	Mat& masksi = masks[i];
    	Mat& imagesi = images[i];
        masksi.create(imagesi.size(), CV_8U);
        masksi.setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator = new cv::PlaneWarper();

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < NumElts; ++i)
    {
        Mat_<float> K;
        CameraParams& camerasi = cameras[i];
        camerasi.K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        Mat& imagesi = images[i];
        Mat& images_warpedi = images_warped[i];
        corners[i] = warper->warp(imagesi, K, camerasi.R, INTER_LINEAR, BORDER_REFLECT, images_warpedi);
        sizes[i] = images_warpedi.size();
        Mat& masksi = masks[i];
        Mat& masks_warpedi = masks_warped[i];
        warper->warp(masksi, K, camerasi.R, INTER_NEAREST, BORDER_CONSTANT, masks_warpedi);
    }

    vector<Mat> images_warped_f(NumElts);
    for (int i = 0; i < NumElts; ++i)
    {
    	Mat& images_warped_fi = images_warped_f[i];
    	Mat& images_warpedi = images_warped[i];
        images_warpedi.convertTo(images_warped_fi, CV_32F);
    }

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);
    errorCode.at<double>(0, 0) = 80;

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "gc_color")
    {
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    double compose_work_aspect = 1;
    errorCode.at<double>(0, 0) = 85;

    for (int img_idx = 0; img_idx < NumElts; ++img_idx)
    {
        // Read image and resize it if necessary
        full_img = MatImages[img_names[img_idx]];
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < NumElts; ++i)
            {
                // Update intrinsics
            	CameraParams& camerasi = cameras[i];
                camerasi.focal *= compose_work_aspect;
                camerasi.ppx *= compose_work_aspect;
                camerasi.ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                	Size& sizei = full_img_sizes[i];
                    sz.width = cvRound(sizei.width * compose_scale);
                    sz.height = cvRound(sizei.height * compose_scale);
                }

                Mat K;

                camerasi.K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, camerasi.R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        CameraParams& camerask =  cameras[img_idx];
        camerask.K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, camerask.R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, camerask.R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        Point& cornersk = corners[img_idx];
        compensator->apply(img_idx, cornersk, img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        Mat& masks_warpedk = masks_warped[img_idx];
        dilate(masks_warpedk, dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f/blend_width);
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, cornersk);
    }

    errorCode.at<double>(0, 0) = 90;

    Mat result, result_mask;
    blender->blend(result, result_mask);
    imwrite("/storage/emulated/0/DCIM/Camera/result.jpg", result);
    errorCode.at<double>(0, 0) = 100;
}
}
