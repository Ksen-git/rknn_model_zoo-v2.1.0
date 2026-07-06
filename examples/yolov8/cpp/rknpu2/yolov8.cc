// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

#include "yolov8.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    app_ctx->rknn_ctx = ctx;

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov8_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ==================== 原同步推理函数（保留但弃用） ====================
int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    printf("WARNING: inference_yolov8_model is deprecated, use pipeline instead.\n");
    return -1;
}

// ==================== 流水线函数实现 ====================

int preprocess_frame(rknn_app_context_t *app_ctx, image_buffer_t *src_img, FrameData *data)
{
    if (!app_ctx || !src_img || !data)
        return -1;

    memset(data, 0, sizeof(FrameData));
    data->frame_id = 0;
    data->src_img = *src_img;

    data->dst_img.width = app_ctx->model_width;
    data->dst_img.height = app_ctx->model_height;
    data->dst_img.format = IMAGE_FORMAT_RGB888;
    data->dst_img.size = get_image_size(&data->dst_img);
    data->dst_img.virt_addr = (unsigned char *)malloc(data->dst_img.size);
    if (!data->dst_img.virt_addr)
    {
        printf("preprocess_frame: malloc dst_img failed\n");
        return -1;
    }

    int ret = convert_image_with_letterbox(src_img, &data->dst_img, &data->letter_box, 114);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        free(data->dst_img.virt_addr);
        data->dst_img.virt_addr = NULL;
        return -1;
    }

    data->inputs[0].index = 0;
    data->inputs[0].type = RKNN_TENSOR_UINT8;
    data->inputs[0].fmt = RKNN_TENSOR_NHWC;
    data->inputs[0].size = data->dst_img.size;
    data->inputs[0].buf = data->dst_img.virt_addr;

    data->valid = true;
    data->processed = false;
    data->outputs = NULL;
    data->output_count = 0;
    return 0;
}

int inference_frame(rknn_app_context_t *app_ctx, FrameData *data)
{
    if (!app_ctx || !data || !data->valid)
        return -1;

    int ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, data->inputs);
    if (ret < 0)
    {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    int num_outputs = app_ctx->io_num.n_output;
    rknn_output *outputs = (rknn_output *)malloc(num_outputs * sizeof(rknn_output));
    if (!outputs)
    {
        printf("malloc outputs failed\n");
        return -1;
    }
    memset(outputs, 0, num_outputs * sizeof(rknn_output));
    for (int i = 0; i < num_outputs; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(app_ctx->rknn_ctx, num_outputs, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        free(outputs);
        return -1;
    }

    data->output_count = num_outputs;
    data->outputs = (rknn_output *)malloc(num_outputs * sizeof(rknn_output));
    if (!data->outputs)
    {
        printf("malloc data->outputs failed\n");
        rknn_outputs_release(app_ctx->rknn_ctx, num_outputs, outputs);
        free(outputs);
        return -1;
    }
    memset(data->outputs, 0, num_outputs * sizeof(rknn_output));

    for (int i = 0; i < num_outputs; i++)
    {
        data->outputs[i].index = i;
        data->outputs[i].want_float = outputs[i].want_float;
        data->outputs[i].size = outputs[i].size;
        data->outputs[i].buf = malloc(outputs[i].size);
        if (!data->outputs[i].buf)
        {
            printf("malloc output buffer %d failed\n", i);
            for (int j = 0; j < i; j++)
                free(data->outputs[j].buf);
            free(data->outputs);
            data->outputs = NULL;
            rknn_outputs_release(app_ctx->rknn_ctx, num_outputs, outputs);
            free(outputs);
            return -1;
        }
        memcpy(data->outputs[i].buf, outputs[i].buf, outputs[i].size);
    }

    rknn_outputs_release(app_ctx->rknn_ctx, num_outputs, outputs);
    free(outputs);

    data->processed = false;
    return 0;
}

int postprocess_frame(rknn_app_context_t *app_ctx, FrameData *data, object_detect_result_list *results)
{
    if (!app_ctx || !data || !data->valid || !data->outputs)
        return -1;

    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    post_process(app_ctx, (void *)data->outputs, &data->letter_box, box_conf_threshold, nms_threshold, results);

    if (results)
    {
        memcpy(&data->results, results, sizeof(object_detect_result_list));
    }

    data->processed = true;
    return 0;
}
