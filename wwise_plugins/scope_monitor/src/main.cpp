// scope_monitor: reads SignalScope's shared-memory capture buffers and shows
// a waveform, spectrogram and spectral-envelope view for whichever instance
// is selected. See ../../mi_common/signal_scope_shm.h for the shared layout
// this reads (read-only -- this process never creates a segment).
#include "signal_scope_shm.h"
#include "fft.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
// GL type/function declarations, via imgui's bundled loader -- pulled in
// explicitly (rather than a system <GL/gl.h>) so the legacy 1.1 calls below
// (glGenTextures, glTexImage2D, ...) resolve to the same dynamically-loaded
// pointers imgui_impl_opengl3.cpp initializes in ImGui_ImplOpenGL3_Init().
#include <backends/imgui_impl_opengl3_loader.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>

namespace {

constexpr int kFftSize = 2048;               // power of two, required by fft::Transform
constexpr int kWaveformFrames = 4096;
constexpr int kSpecWidth = 256;               // scrolling history, one column per push
constexpr int kSpecHeight = 256;              // frequency bins, linearly binned 0..Nyquist
constexpr float kMinDb = -80.0f;
constexpr float kMaxDb = 0.0f;
constexpr double kSpecPushIntervalSec = 1.0 / 30.0;

float ToDb(float magnitude)
{
    return 20.0f * std::log10(std::max(magnitude, 1e-6f));
}

// Small hand-rolled heat colormap (black -> blue -> green -> yellow -> white),
// good enough to read relative energy without pulling in a palette library.
void HeatColor(float t, unsigned char out[3])
{
    t = std::clamp(t, 0.0f, 1.0f);
    struct Stop { float t; float r, g, b; };
    static const Stop stops[] = {
        { 0.00f, 0.00f, 0.00f, 0.05f },
        { 0.25f, 0.05f, 0.05f, 0.55f },
        { 0.50f, 0.10f, 0.55f, 0.45f },
        { 0.75f, 0.90f, 0.75f, 0.10f },
        { 1.00f, 1.00f, 1.00f, 0.90f },
    };
    int i = 0;
    while (i < 3 && t > stops[i + 1].t)
        ++i;
    const Stop& a = stops[i];
    const Stop& b = stops[i + 1];
    const float span = (b.t - a.t) > 1e-6f ? (b.t - a.t) : 1.0f;
    const float f = (t - a.t) / span;
    out[0] = (unsigned char)std::clamp((a.r + (b.r - a.r) * f) * 255.0f, 0.0f, 255.0f);
    out[1] = (unsigned char)std::clamp((a.g + (b.g - a.g) * f) * 255.0f, 0.0f, 255.0f);
    out[2] = (unsigned char)std::clamp((a.b + (b.b - a.b) * f) * 255.0f, 0.0f, 255.0f);
}

// Per-instance display state: the FFT window, waveform scratch buffer and the
// scrolling spectrogram texture. Rebuilt whenever the user picks a different
// scope or the ring's generation counter changes (the writer re-opened it).
struct ScopeView
{
    mi::scope::RingReader ring;
    uint32_t scopeId = 0xFFFFFFFFu;
    uint64_t boundGeneration = 0;
    int channel = 0;

    std::vector<float> hann;
    std::vector<float> waveformBuf;
    std::vector<float> fftInputBuf;
    std::vector<float> magnitude; // linear, size kFftSize/2+1

    std::vector<float> specDb; // row-major, kSpecHeight rows x kSpecWidth cols, oldest column at 0
    std::vector<unsigned char> specPixels; // RGBA
    GLuint specTex = 0;
    double lastSpecPushTime = 0.0;

    bool Bind(uint32_t id)
    {
        if (!ring.Open(id))
            return false;
        scopeId = id;
        boundGeneration = 0; // force a reset below so buffers match the (possibly new) channel count
        channel = 0;

        hann.resize(kFftSize);
        fft::HannWindow(hann);
        waveformBuf.assign(kWaveformFrames, 0.0f);
        fftInputBuf.assign(kFftSize, 0.0f);
        magnitude.assign(kFftSize / 2 + 1, 0.0f);
        specDb.assign(size_t(kSpecWidth) * kSpecHeight, kMinDb);
        specPixels.assign(size_t(kSpecWidth) * kSpecHeight * 4, 0);
        lastSpecPushTime = 0.0;

        if (specTex == 0)
            glGenTextures(1, &specTex);
        glBindTexture(GL_TEXTURE_2D, specTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return true;
    }

    void Unbind()
    {
        ring.Close();
        if (specTex)
        {
            glDeleteTextures(1, &specTex);
            specTex = 0;
        }
        scopeId = 0xFFFFFFFFu;
    }

    void PushSpectrogramColumn()
    {
        // Bin the linear FFT magnitude spectrum down into kSpecHeight rows
        // (row 0 = DC/low frequency) and scroll the image left by one column.
        std::vector<float> col(kSpecHeight, kMinDb);
        const int nBins = (int)magnitude.size();
        for (int row = 0; row < kSpecHeight; ++row)
        {
            const int lo = row * nBins / kSpecHeight;
            const int hi = std::max(lo + 1, (row + 1) * nBins / kSpecHeight);
            float peak = 0.0f;
            for (int b = lo; b < hi && b < nBins; ++b)
                peak = std::max(peak, magnitude[b]);
            col[row] = std::clamp(ToDb(peak), kMinDb, kMaxDb);
        }

        for (int row = 0; row < kSpecHeight; ++row)
        {
            float* rowPtr = &specDb[size_t(row) * kSpecWidth];
            std::memmove(rowPtr, rowPtr + 1, (kSpecWidth - 1) * sizeof(float));
            rowPtr[kSpecWidth - 1] = col[row];
        }

        for (int row = 0; row < kSpecHeight; ++row)
        {
            for (int c = 0; c < kSpecWidth; ++c)
            {
                const float db = specDb[size_t(row) * kSpecWidth + c];
                const float t = (db - kMinDb) / (kMaxDb - kMinDb);
                // Image row 0 at the top should be the highest frequency.
                const int destRow = kSpecHeight - 1 - row;
                unsigned char* px = &specPixels[(size_t(destRow) * kSpecWidth + c) * 4];
                HeatColor(t, px);
                px[3] = 255;
            }
        }

        glBindTexture(GL_TEXTURE_2D, specTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kSpecWidth, kSpecHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, specPixels.data());
    }

    void Update(double nowSec)
    {
        if (!ring.IsOpen())
            return;
        if (channel >= (int)ring.NumChannels())
            channel = 0;

        ring.ReadLatest((uint32_t)channel, waveformBuf.data(), (uint32_t)waveformBuf.size());
        ring.ReadLatest((uint32_t)channel, fftInputBuf.data(), (uint32_t)fftInputBuf.size());
        fft::MagnitudeSpectrum(fftInputBuf.data(), hann, magnitude);

        if (nowSec - lastSpecPushTime >= kSpecPushIntervalSec)
        {
            PushSpectrogramColumn();
            lastSpecPushTime = nowSec;
        }
    }
};

void DrawWaveform(const ScopeView& view)
{
    if (view.waveformBuf.empty())
        return;
    ImGui::PlotLines("##waveform", view.waveformBuf.data(), (int)view.waveformBuf.size(),
        0, nullptr, -1.0f, 1.0f, ImVec2(-1, 150));
}

void DrawSpectralEnvelope(const ScopeView& view)
{
    if (view.magnitude.empty() || view.ring.SampleRate() == 0)
        return;

    ImGui::BeginChild("##envelope", ImVec2(-1, 180), true);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 size = ImGui::GetContentRegionAvail();
    if (size.x > 1 && size.y > 1)
    {
        const float nyquist = float(view.ring.SampleRate()) * 0.5f;
        const float minHz = 20.0f;
        const float logMin = std::log10(minHz);
        const float logMax = std::log10(std::max(nyquist, minHz * 2.0f));
        const int nBins = (int)view.magnitude.size();
        const float binHz = nyquist / float(nBins - 1);

        std::vector<ImVec2> points;
        points.reserve(nBins);
        for (int i = 1; i < nBins; ++i) // skip DC, undefined on a log axis
        {
            const float hz = std::max(binHz * i, minHz);
            const float logHz = std::log10(hz);
            const float x = (logHz - logMin) / (logMax - logMin);
            if (x < 0.0f || x > 1.0f)
                continue;
            const float db = std::clamp(ToDb(view.magnitude[i]), kMinDb, kMaxDb);
            const float y = 1.0f - (db - kMinDb) / (kMaxDb - kMinDb);
            points.push_back(ImVec2(origin.x + x * size.x, origin.y + y * size.y));
        }
        if (points.size() > 1)
            draw->AddPolyline(points.data(), (int)points.size(), IM_COL32(90, 200, 250, 255), 0, 1.5f);
        draw->AddRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), IM_COL32(90, 90, 90, 255));
    }
    ImGui::Dummy(size);
    ImGui::EndChild();
}

void DrawSpectrogram(ScopeView& view)
{
    if (view.specTex == 0)
        return;
    // Unlike PlotLines etc., ImGui::Image() takes image_size literally -- it
    // does not treat a negative width as "fill available space" -- so the
    // available width has to be resolved explicitly here.
    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::Image((ImTextureID)(intptr_t)view.specTex, ImVec2(width, 220));
}

} // namespace

int main()
{
    if (!glfwInit())
    {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1100, 800, "SignalScope Monitor", nullptr, nullptr);
    if (!window)
    {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    mi::scope::DirectoryReader dirReader;
    ScopeView view;
    int selectedIndex = -1;

    const auto t0 = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        const double nowSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<mi::scope::ScopeInfo> scopes = dirReader.ListActive();
        std::sort(scopes.begin(), scopes.end(), [](const auto& a, const auto& b) { return a.scopeId < b.scopeId; });

        // If the previously bound scope disappeared or its writer re-opened
        // (generation bump), drop the binding so the panel below re-creates it.
        if (view.ring.IsOpen())
        {
            const auto it = std::find_if(scopes.begin(), scopes.end(),
                [&](const mi::scope::ScopeInfo& s) { return s.scopeId == view.scopeId; });
            if (it == scopes.end() || it->generation != view.boundGeneration)
            {
                view.Unbind();
                selectedIndex = -1;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("SignalScope Monitor", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);

        ImGui::Text("Active scopes: %d", (int)scopes.size());
        ImGui::SameLine();
        if (scopes.empty())
            ImGui::TextDisabled("(insert a Signal Scope effect in Wwise and enter Play/Preview)");

        std::vector<std::string> labels;
        labels.reserve(scopes.size());
        for (const auto& s : scopes)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "#%u  %s  (%uch, %uHz)",
                s.scopeId, s.label.empty() ? "Scope" : s.label.c_str(), s.numChannels, s.sampleRate);
            labels.push_back(buf);
        }

        std::vector<const char*> labelPtrs;
        labelPtrs.reserve(labels.size());
        for (auto& l : labels)
            labelPtrs.push_back(l.c_str());

        ImGui::SetNextItemWidth(400);
        if (ImGui::Combo("Scope", &selectedIndex, labelPtrs.empty() ? nullptr : labelPtrs.data(), (int)labelPtrs.size()))
        {
            if (selectedIndex >= 0 && selectedIndex < (int)scopes.size())
            {
                view.Unbind();
                view.Bind(scopes[selectedIndex].scopeId);
                view.boundGeneration = scopes[selectedIndex].generation;
            }
        }

        if (view.ring.IsOpen())
        {
            view.Update(nowSec);

            if (view.ring.NumChannels() > 1)
            {
                std::vector<std::string> chLabels;
                std::vector<const char*> chLabelPtrs;
                chLabels.reserve(view.ring.NumChannels());
                for (uint32_t c = 0; c < view.ring.NumChannels(); ++c)
                {
                    const char* name = view.ring.ChannelName(c);
                    char buf[32];
                    if (name[0] != '\0')
                        std::snprintf(buf, sizeof(buf), "%u: %s", c, name);
                    else
                        std::snprintf(buf, sizeof(buf), "Ch %u", c);
                    chLabels.push_back(buf);
                }
                for (auto& l : chLabels)
                    chLabelPtrs.push_back(l.c_str());

                ImGui::SameLine();
                ImGui::SetNextItemWidth(140);
                ImGui::Combo("Channel", &view.channel, chLabelPtrs.data(), (int)chLabelPtrs.size());
            }

            ImGui::Separator();
            ImGui::Text("Waveform");
            DrawWaveform(view);

            ImGui::Text("Spectral envelope (20Hz - Nyquist, log freq / dB)");
            DrawSpectralEnvelope(view);

            ImGui::Text("Spectrogram (scrolling, ~%.0fs window)", kSpecWidth * kSpecPushIntervalSec);
            DrawSpectrogram(view);
        }
        else
        {
            ImGui::Separator();
            ImGui::TextDisabled("Select a scope above to view it.");
        }

        ImGui::End();

        ImGui::Render();
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    view.Unbind();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
