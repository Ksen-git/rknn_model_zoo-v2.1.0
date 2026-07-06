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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103)
#include "dma_alloc.hpp"
#endif

template <typename T>
class SafeQueue
{
public:
    void push(T item)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(item);
        cond_.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [this]()
                   { return !queue_.empty() || stop_; });
        if (stop_ && queue_.empty())
            return T();
        T item = queue_.front();
        queue_.pop();
        return item;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
        cond_.notify_all();
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cond_;
    bool stop_ = false;
};

std::mutex g_timer_mutex;
double g_pre_total_ms = 0.0;
double g_inf_total_ms = 0.0;
double g_post_total_ms = 0.0;
int g_frame_count = 0;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Usage: %s <model_path> <image_path> <loop_count>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const int LOOP_COUNT = atoi(argv[3]);
    if (LOOP_COUNT <= 0)
        return -1;

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // ===== 所有变量提前声明 =====
    image_buffer_t template_img;
    memset(&template_img, 0, sizeof(image_buffer_t));
    SafeQueue<FrameData *> pre_queue;
    SafeQueue<FrameData *> inf_queue;
    std::atomic<bool> running(true);
    std::atomic<int> frame_counter(0);
    std::atomic<int> processed_counter(0);
    int total_frames = 0;
    std::thread pre_thread, inf_thread, post_thread;

    // 提前声明计时变量（关键！避免goto跨初始化）
    std::chrono::time_point<std::chrono::high_resolution_clock> t_pipeline_start, t_pipeline_end;
    double pipeline_total_ms = 0.0;

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail!\n");
        goto out;
    }

    ret = read_image(image_path, &template_img);
    if (ret != 0)
    {
        printf("read image fail!\n");
        goto out;
    }
    free(template_img.virt_addr);
    template_img.virt_addr = NULL;

    t_pipeline_start = std::chrono::high_resolution_clock::now();

    // 预处理线程
    pre_thread = std::thread([&]()
                             {
        while (running) {
            int id = frame_counter.fetch_add(1);
            if (id >= LOOP_COUNT) break;
            image_buffer_t src_img;
            memset(&src_img, 0, sizeof(image_buffer_t));
            if (read_image(image_path, &src_img) != 0) break;
            FrameData *data = new FrameData();
            auto t_start = std::chrono::high_resolution_clock::now();
            if (preprocess_frame(&rknn_app_ctx, &src_img, data) != 0) { delete data; free(src_img.virt_addr); break; }
            data->frame_id = id;
            free(src_img.virt_addr);
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            { std::lock_guard<std::mutex> lock(g_timer_mutex); g_pre_total_ms += elapsed_ms; }
            pre_queue.push(data);
        }
        pre_queue.stop(); });

    // 推理线程
    inf_thread = std::thread([&]()
                             {
        while (running) {
            FrameData *data = pre_queue.pop();
            if (!data) break;
            auto t_start = std::chrono::high_resolution_clock::now();
            if (inference_frame(&rknn_app_ctx, data) != 0) { free(data->dst_img.virt_addr); delete data; continue; }
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            { std::lock_guard<std::mutex> lock(g_timer_mutex); g_inf_total_ms += elapsed_ms; }
            inf_queue.push(data);
        }
        inf_queue.stop(); });

    // 后处理线程
    post_thread = std::thread([&]()
                              {
        object_detect_result_list results;
        while (running) {
            FrameData *data = inf_queue.pop();
            if (!data) break;
            auto t_start = std::chrono::high_resolution_clock::now();
            if (postprocess_frame(&rknn_app_ctx, data, &results) != 0) {
                free(data->dst_img.virt_addr);
                if (data->outputs) { for (int i = 0; i < data->output_count; i++) free(data->outputs[i].buf); free(data->outputs); }
                delete data; continue;
            }
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            { std::lock_guard<std::mutex> lock(g_timer_mutex); g_post_total_ms += elapsed_ms; g_frame_count++; }
            free(data->dst_img.virt_addr);
            if (data->outputs) { for (int i = 0; i < data->output_count; i++) free(data->outputs[i].buf); free(data->outputs); }
            delete data;
            int done = processed_counter.fetch_add(1) + 1;
            if (done % 10 == 0 || done == LOOP_COUNT) printf("  Processed %d/%d frames\n", done, LOOP_COUNT);
        } });

    pre_thread.join();
    inf_thread.join();
    post_thread.join();

    t_pipeline_end = std::chrono::high_resolution_clock::now();
    pipeline_total_ms = std::chrono::duration<double, std::milli>(t_pipeline_end - t_pipeline_start).count();

    total_frames = g_frame_count;
    if (total_frames > 0)
    {
        double avg_pre = g_pre_total_ms / total_frames;
        double avg_inf = g_inf_total_ms / total_frames;
        double avg_post = g_post_total_ms / total_frames;
        double avg_total = avg_pre + avg_inf + avg_post;
        double total_all = g_pre_total_ms + g_inf_total_ms + g_post_total_ms;
        double real_fps = total_frames / (pipeline_total_ms / 1000.0);
        printf("\n============================================================\n");
        printf("  YOLOv8 Pipeline Benchmark Results (%d frames)\n", total_frames);
        printf("============================================================\n");
        printf("  Stage               Total(ms)     Avg(ms)       Share\n");
        printf("  --------------------------------------------------------\n");
        printf("  Preprocess          %10.2f      %8.2f       %5.1f%%\n", g_pre_total_ms, avg_pre, (g_pre_total_ms / total_all) * 100);
        printf("  Inference           %10.2f      %8.2f       %5.1f%%\n", g_inf_total_ms, avg_inf, (g_inf_total_ms / total_all) * 100);
        printf("  Postprocess         %10.2f      %8.2f       %5.1f%%\n", g_post_total_ms, avg_post, (g_post_total_ms / total_all) * 100);
        printf("  --------------------------------------------------------\n");
        printf("  Total               %10.2f      %8.2f      100.0%%\n", total_all, avg_total);
        printf("\n  Pipeline wall time:  %.2f ms\n", pipeline_total_ms);
        printf("  Average per frame:   %.2f ms\n", pipeline_total_ms / total_frames);
        printf("  Real FPS:            %.1f\n", real_fps);
        printf("============================================================\n");
    }

out:
    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);
    if (template_img.virt_addr != NULL)
        free(template_img.virt_addr);
    if (pre_thread.joinable())
        pre_thread.join();
    if (inf_thread.joinable())
        inf_thread.join();
    if (post_thread.joinable())
        post_thread.join();
    return 0;
}
