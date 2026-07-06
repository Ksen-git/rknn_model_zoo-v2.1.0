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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <chrono>

extern double g_preprocess_ms;
extern double g_inference_ms;
extern double g_postprocess_ms;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // ===== 所有变量提前声明（避免goto问题） =====
    image_buffer_t src_image;
    object_detect_result_list od_results;
    const int LOOP_COUNT = 100;
    double avg_pre, avg_inf, avg_post, total_all, avg_total;
    double total_wall_ms; // 总耗时(毫秒)
    double real_fps;      // 帧率
    char text[256];

    // 总计时变量
    std::chrono::time_point<std::chrono::high_resolution_clock> total_start, total_end;
    // ===========================================

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    // ===== Warm-up 5 times =====
    printf("\nWarm-up 5 times...\n");
    for (int w = 0; w < 5; w++)
    {
        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail at warmup %d! ret=%d\n", w, ret);
            goto out;
        }
    }
    printf("Warm-up done.\n");

    // Reset accumulators
    g_preprocess_ms = 0.0;
    g_inference_ms = 0.0;
    g_postprocess_ms = 0.0;

    // ===== Benchmark: 100 loops =====
    printf("\nBenchmark: running %d loops...\n", LOOP_COUNT);

    // 记录总开始时间
    total_start = std::chrono::high_resolution_clock::now();

    for (int loop = 0; loop < LOOP_COUNT; loop++)
    {
        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail at loop %d! ret=%d\n", loop, ret);
            goto out;
        }
        if ((loop + 1) % 10 == 0)
        {
            printf("  %d/%d done\n", loop + 1, LOOP_COUNT);
        }
    }

    // 记录总结束时间
    total_end = std::chrono::high_resolution_clock::now();
    total_wall_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // ===== 计算平均值和帧率 =====
    avg_pre = g_preprocess_ms / LOOP_COUNT;
    avg_inf = g_inference_ms / LOOP_COUNT;
    avg_post = g_postprocess_ms / LOOP_COUNT;
    total_all = g_preprocess_ms + g_inference_ms + g_postprocess_ms;
    avg_total = avg_pre + avg_inf + avg_post;
    real_fps = LOOP_COUNT / (total_wall_ms / 1000.0); // 帧数 / 总耗时(秒)

    // ===== 打印结果 =====
    printf("\n");
    printf("============================================================\n");
    printf("  YOLOv8 Benchmark Results (%d loops)\n", LOOP_COUNT);
    printf("============================================================\n");
    printf("  Stage               Total(ms)     Avg(ms)       Share\n");
    printf("  --------------------------------------------------------\n");
    printf("  Preprocess          %10.2f      %8.2f       %5.1f%%\n",
           g_preprocess_ms, avg_pre, (g_preprocess_ms / total_all) * 100.0);
    printf("  Inference           %10.2f      %8.2f       %5.1f%%\n",
           g_inference_ms, avg_inf, (g_inference_ms / total_all) * 100.0);
    printf("  Postprocess         %10.2f      %8.2f       %5.1f%%\n",
           g_postprocess_ms, avg_post, (g_postprocess_ms / total_all) * 100.0);
    printf("  --------------------------------------------------------\n");
    printf("  Total               %10.2f      %8.2f      100.0%%\n",
           total_all, avg_total);
    printf("\n  Wall time:          %.2f ms\n", total_wall_ms);
    printf("  Average per frame:   %.2f ms\n", total_wall_ms / LOOP_COUNT);
    printf("  Real FPS:            %.1f\n", real_fps);
    printf("============================================================\n");

    // ===== Draw detection boxes =====
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    write_image("out.png", &src_image);
    printf("\nResult saved to out.png\n");
    printf("============================================================\n");

out:
    deinit_post_process();

    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
