#include <libfreenect/libfreenect.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sys/time.h>
#include <thread>

static const int DW = 640, DH = 480;

static std::mutex        s_depth_mtx, s_video_mtx;
static uint8_t           s_depth_pix[DW * DH * 3]{};
static uint8_t           s_video_pix[DW * DH * 3]{};
// Two video buffers: freenect writes into s_video_back; callback swaps them.
static uint8_t           s_video_back[DW * DH * 3]{};
static std::atomic<bool> s_depth_dirty{false}, s_video_dirty{false};
static std::atomic<bool> s_running{true};
static GLFWwindow*       s_win = nullptr;

static void sig_handler(int) {
    if (s_win) glfwSetWindowShouldClose(s_win, GLFW_TRUE);
}

static void depth_cb(freenect_device*, void* buf, uint32_t) {
    auto* d = static_cast<uint16_t*>(buf);
    std::lock_guard<std::mutex> lk(s_depth_mtx);
    for (int i = 0; i < DW * DH; i++) {
        uint32_t raw = d[i] & 0x7FFu;
        // closer = brighter; no-data (2047) = black
        uint8_t v = (raw == 2047u) ? 0u : static_cast<uint8_t>((2047u - raw) * 255u / 2047u);
        s_depth_pix[3*i+0] = v;
        s_depth_pix[3*i+1] = v;
        s_depth_pix[3*i+2] = v;
    }
    s_depth_dirty = true;
}

static void video_cb(freenect_device*, void* buf, uint32_t) {
    std::lock_guard<std::mutex> lk(s_video_mtx);
    memcpy(s_video_pix, buf, DW * DH * 3);
    s_video_dirty = true;
}

static void freenect_thread_fn(freenect_context* ctx) {
    timeval tv{0, 100'000};
    while (s_running)
        freenect_process_events_timeout(ctx, &tv);
}

static GLuint make_tex() {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, DW, DH, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

static void upload_tex(GLuint t, const uint8_t* rgb) {
    glBindTexture(GL_TEXTURE_2D, t);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DW, DH, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glBindTexture(GL_TEXTURE_2D, 0);
}

int main() {
    freenect_context* ctx = nullptr;
    if (freenect_init(&ctx, nullptr) < 0) {
        fprintf(stderr, "freenect_init failed\n");
        return 1;
    }
    freenect_set_log_level(ctx, FREENECT_LOG_WARNING);

    if (freenect_num_devices(ctx) < 1) {
        fprintf(stderr, "No Kinect devices found\n");
        freenect_shutdown(ctx);
        return 1;
    }

    freenect_device* dev = nullptr;
    for (int i = 0; i < 5; i++) {
        if (i > 0) {
            // Full teardown + reinit to clear bad USB state from the failed attempt
            freenect_shutdown(ctx);
            ctx = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            if (freenect_init(&ctx, nullptr) < 0) break;
            freenect_set_log_level(ctx, FREENECT_LOG_WARNING);
            if (freenect_num_devices(ctx) < 1) continue;
        }
        if (freenect_open_device(ctx, &dev, 0) == 0) break;
        fprintf(stderr, "Open attempt %d/5 failed, retrying...\n", i + 1);
        dev = nullptr;
    }
    if (!dev || !ctx) {
        fprintf(stderr, "Failed to open Kinect device\n");
        if (ctx) freenect_shutdown(ctx);
        return 1;
    }

    freenect_set_depth_callback(dev, depth_cb);
    freenect_set_video_callback(dev, video_cb);
    freenect_set_depth_mode(dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_video_mode(dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    freenect_set_video_buffer(dev, s_video_back);
    if (freenect_start_depth(dev) < 0) fprintf(stderr, "freenect_start_depth failed\n");
    if (freenect_start_video(dev) < 0) fprintf(stderr, "freenect_start_video failed\n");

    std::thread fk_thread(freenect_thread_fn, ctx);

    glfwSetErrorCallback([](int e, const char* m) { fprintf(stderr, "GLFW %d: %s\n", e, m); });
    if (!glfwInit()) { s_running = false; fk_thread.join(); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* win = glfwCreateWindow(1320, 560, "Kinect Viewer", nullptr, nullptr);
    if (!win) { glfwTerminate(); s_running = false; fk_thread.join(); return 1; }
    s_win = win;
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    GLuint depth_tex = make_tex();
    GLuint video_tex = make_tex();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (s_depth_dirty.exchange(false)) {
            std::lock_guard<std::mutex> lk(s_depth_mtx);
            upload_tex(depth_tex, s_depth_pix);
        }
        if (s_video_dirty.exchange(false)) {
            std::lock_guard<std::mutex> lk(s_video_mtx);
            upload_tex(video_tex, s_video_pix);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("Depth", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Image((ImTextureID)(uintptr_t)depth_tex, {DW, DH});
        ImGui::End();

        ImGui::SetNextWindowPos({DW + 16.f, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("Video", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Image((ImTextureID)(uintptr_t)video_tex, {DW, DH});
        ImGui::End();

        ImGui::Render();
        int fw, fh;
        glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // Stop streams while the event thread is still running — stop commands
    // are USB control transfers that need the event loop to complete.
    freenect_stop_depth(dev);
    freenect_stop_video(dev);
    s_running = false;
    fk_thread.join();

    freenect_close_device(dev);
    freenect_shutdown(ctx);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
