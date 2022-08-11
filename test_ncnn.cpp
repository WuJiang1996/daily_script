#include <string>
#include <vector>
#include "iostream"
//#include <fstream>
//#include < ctime >
//#include <direct.h>
//#include <io.h>

// ncnn
#include "ncnn/layer.h"
#include "ncnn/net.h"
#include "ncnn/benchmark.h"
//#include "gpu.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net yolov5;

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object &a, const Object &b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }
    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob, float prob_threshold, std::vector<Object> &objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)

        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;
                    objects.push_back(obj);
                }
            }
        }
    }
}

extern "C"
{

    void release()
    {
        fprintf(stderr, "YoloV5Ncnn finished!");

        // ncnn::destroy_gpu_instance();
    }

    int init()
    {
        fprintf(stderr, "YoloV5Ncnn init!\n");
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
        opt.use_packing_layout = true;

        yolov5.opt = opt;

        yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
        // init param
        {
            int ret = yolov5.load_param("yolov5s.param");
            if (ret != 0)
            {
                std::cout << "ret= " << ret << std::endl;
                fprintf(stderr, "YoloV5Ncnn, load_param failed");
                return -301;
            }
        }

        // init bin
        {
            int ret = yolov5.load_model("yolov5s.bin");
            if (ret != 0)
            {
                fprintf(stderr, "YoloV5Ncnn, load_model failed");
                return -301;
            }
        }
        return 0;
    }

    int detect(cv::Mat img, std::vector<Object> &objects)
    {

        double start_time = ncnn::get_current_time();
        const int target_size = 320;

        // letterbox pad to multiple of 32
        const int width = img.cols;  // 1280
        const int height = img.rows; // 720
        int w = img.cols;            // 1280
        int h = img.rows;            // 720
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w; // 640/1280
            w = target_size;                // 640
            h = h * scale;                  // 360
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }
        cv::resize(img, img, cv::Size(w, h));
        ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);

        // pad to target_size rectangle
        // yolov5/utils/datasets.py letterbox
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
        // yolov5
        // std::vector<Object> objects;
        {
            const float prob_threshold = 0.4f;
            const float nms_threshold = 0.51f;

            const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
            in_pad.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = yolov5.create_extractor();
            // ex.set_vulkan_compute(use_gpu);

            ex.input("images", in_pad);
            std::vector<Object> proposals;

            // anchor setting from yolov5/models/yolov5s.yaml

            // stride 8
            {
                ncnn::Mat out;
                ex.extract("output", out);
                ncnn::Mat anchors(6);
                anchors[0] = 10.f;
                anchors[1] = 13.f;
                anchors[2] = 16.f;
                anchors[3] = 30.f;
                anchors[4] = 33.f;
                anchors[5] = 23.f;

                std::vector<Object> objects8;
                generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
                proposals.insert(proposals.end(), objects8.begin(), objects8.end());
            }

            // stride 16
            {
                ncnn::Mat out;
                ex.extract("771", out);

                ncnn::Mat anchors(6);
                anchors[0] = 30.f;
                anchors[1] = 61.f;
                anchors[2] = 62.f;
                anchors[3] = 45.f;
                anchors[4] = 59.f;
                anchors[5] = 119.f;

                std::vector<Object> objects16;
                generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

                proposals.insert(proposals.end(), objects16.begin(), objects16.end());
            }
            // stride 32
            {
                ncnn::Mat out;
                ex.extract("791", out);
                ncnn::Mat anchors(6);
                anchors[0] = 116.f;
                anchors[1] = 90.f;
                anchors[2] = 156.f;
                anchors[3] = 198.f;
                anchors[4] = 373.f;

                anchors[5] = 326.f;

                std::vector<Object> objects32;
                generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

                proposals.insert(proposals.end(), objects32.begin(), objects32.end());
            }

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);
            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_threshold);

            int count = picked.size();
            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].x - (wpad / 2)) / scale;
                float y0 = (objects[i].y - (hpad / 2)) / scale;
                float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
                float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
                objects[i].x = x0;
                objects[i].y = y0;
                objects[i].w = x1;
                objects[i].h = y1;
            }
        }

        return 0;
    }
}

static const char *class_names[] = {
    "four_fingers", "hand_with_fingers_splayed", "index_pointing_up", "little_finger",
    "ok_hand", "raised_fist", "raised_hand", "sign_of_the_horns", "three", "thumbup", "victory_hand"};

void draw_face_box(cv::Mat &bgr, std::vector<Object> object) //主要的emoji显示函数
{
    for (int i = 0; i < object.size(); i++)
    {
        const auto obj = object[i];
        cv::rectangle(bgr, cv::Point(obj.x, obj.y), cv::Point(obj.w, obj.h), cv::Scalar(0, 255, 0), 3, 8, 0);
        std::cout << "label:" << class_names[obj.label] << std::endl;
        string emoji_path = "emoji\\" + string(class_names[obj.label]) + ".png"; //这个是emoji图片的路径
        cv::Mat logo = cv::imread(emoji_path);
        if (logo.empty())
        {
            std::cout << "imread logo failed!!!" << std::endl;
            return;
        }
        resize(logo, logo, cv::Size(80, 80));
        cv::Mat imageROI = bgr(cv::Range(obj.x, obj.x + logo.rows), cv::Range(obj.y, obj.y + logo.cols)); // emoji的图片放在图中的位置，也就是手势框的旁边
        logo.copyTo(imageROI);                                                                            //把emoji放在原图中
    }
}

int main()
{
    Mat frame;

    VideoCapture capture(0);
    init();
    while (true)
    {
        capture >> frame;
        if (!frame.empty())
        {
            std::vector<Object> objects;
            detect(frame, objects);
            draw_face_box(frame, objects);
            imshow("window", frame);
        }
        if (waitKey(20) == 'q')
            break;
    }

    capture.release();

    return 0;
}