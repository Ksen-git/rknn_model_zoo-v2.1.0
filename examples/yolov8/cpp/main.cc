cat > examples / yolov8 / cpp / main.cc << 'EOF'
// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

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

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;
    const int LOOP_COUNT = 100;

    // Warm-up 5 times
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

    // Benchmark
    printf("\nBenchmark: running %d loops...\n", LOOP_COUNT);
    for (int loop = 0; loop < LOOP_COUNT; loop++)
    {
        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail at loop %d! ret=%d\n", loop, ret);
            goto out;
        }
        if ((loop + 1) % 10 == 0)
            printf("  %d/%d done\n", loop + 1, LOOP_COUNT);
    }

    // Calculate averages
    double avg_pre = g_preprocess_ms / LOOP_COUNT;
    double avg_inf = g_inference_ms / LOOP_COUNT;
    double avg_post = g_postprocess_ms / LOOP_COUNT;
    double total_all = g_preprocess_ms + g_inference_ms + g_postprocess_ms;
    double avg_total = avg_pre + avg_inf + avg_post;

    // Print results
    printf("\n============================================================\n");
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
    printf("  Average total time per loop: %.2f ms\n", avg_total);

out:
    deinit_post_process();
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
        printf("release_yolov8_model fail! ret=%d\n", ret);
    if (src_image.virt_addr != NULL)
        free(src_image.virt_addr);
    return 0;
}
EOF
