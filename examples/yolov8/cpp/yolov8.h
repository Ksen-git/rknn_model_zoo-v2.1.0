#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

#include "rknn_api.h"
#include "common.h"

#if defined(RV1106_1103)
typedef struct
{
    char *dma_buf_virt_addr;
    int dma_buf_fd;
    int size;
} rknn_dma_buf;
#endif

typedef struct
{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
#if defined(RV1106_1103)
    rknn_tensor_mem *input_mems[1];
    rknn_tensor_mem *output_mems[9];
    rknn_dma_buf img_dma_buf;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

#include "postprocess.h"

// ===== 流水线帧数据结构 =====
typedef struct
{
    int frame_id;
    image_buffer_t src_img;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[1];
    rknn_output *outputs;
    int output_count;
    bool valid;
    bool processed;
    object_detect_result_list results;
} FrameData;

// ===== 流水线函数声明 =====
int preprocess_frame(rknn_app_context_t *app_ctx, image_buffer_t *src_img, FrameData *data);
int inference_frame(rknn_app_context_t *app_ctx, FrameData *data);
int postprocess_frame(rknn_app_context_t *app_ctx, FrameData *data, object_detect_result_list *results);

int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_yolov8_model(rknn_app_context_t *app_ctx);
int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results);

#endif //_RKNN_DEMO_YOLOV8_H_
