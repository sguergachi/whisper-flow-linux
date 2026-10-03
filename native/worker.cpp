// Private stdin/stdout protocol. No socket and no audio files in the child.
// Request: WFW1, 7 u32 (sample count, beam, best-of, suppress, audio context,
// language bytes, prompt bytes), 2 f32 (temperature, no-speech threshold),
// UTF-8 language + prompt, then mono 16 kHz float32 PCM; all little endian.
// Response: WFR1, u32 byte length, u32 status (0 success), UTF-8 body.
#include "whisper.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

static bool read_exact(void * data, size_t size) {
    auto * bytes = static_cast<unsigned char *>(data);
    while (size) {
        size_t n = std::fread(bytes, 1, size, stdin);
        if (!n) return false;
        bytes += n;
        size -= n;
    }
    return true;
}

static uint32_t u32(const unsigned char * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static float f32(const unsigned char * p) {
    uint32_t bits = u32(p);
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

static std::atomic<bool> gpu_failed{false};
static void engine_log(enum ggml_log_level, const char * text, void *) {
    std::fputs(text, stderr);
    if (std::strstr(text, "whisper_backend_init_gpu: no GPU found") ||
        std::strstr(text, "whisper_backend_init_gpu: failed to initialize"))
        gpu_failed = true;
}

static void reply(const std::string & text, uint32_t status = 0) {
    uint32_t size = uint32_t(text.size());
    unsigned char header[12] = {'W', 'F', 'R', '1'};
    for (int i = 0; i < 4; ++i) {
        header[4 + i] = (size >> (8 * i)) & 255;
        header[8 + i] = (status >> (8 * i)) & 255;
    }
    if (std::fwrite(header, 1, sizeof(header), stdout) != sizeof(header) ||
        std::fwrite(text.data(), 1, text.size(), stdout) != text.size() ||
        std::fflush(stdout)) throw std::runtime_error("response pipe closed");
}

static int run(const std::vector<std::string> & args) {
    if (args.size() == 2 && args[1] == "--help") {
        std::fprintf(stderr, "whisper-flow-worker MODEL THREADS GPU(0/1)\n");
        return 0;
    }
    if (args.size() != 4) return 2;
    int threads = std::stoi(args[2]);
    if (threads < 1 || threads > 64) return 2;
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    auto cp = whisper_context_default_params();
    whisper_log_set(engine_log, nullptr);
    cp.use_gpu = args[3] == "1";
    cp.flash_attn = cp.use_gpu;
    // Select a real GPU before loading a large model. whisper.cpp otherwise
    // silently falls back to CPU, making large-v3-turbo unusably slow.
    if (cp.use_gpu) {
        ggml_backend_load_all();
        int selected = -1, gpu_index = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            // This path is selected on NVIDIA machines. Do not accidentally
            // put their large model on an Intel iGPU enumerated first.
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU && selected < 0) selected = gpu_index;
            if (std::strstr(ggml_backend_dev_description(dev), "NVIDIA")) {
                selected = gpu_index;
                break;
            }
            ++gpu_index;
        }
        if (selected < 0) {
            reply("No Vulkan GPU is available", 1);
            return 3;
        }
        cp.gpu_device = selected;
    }
    // MinGW's upstream file initializer uses a narrow ifstream path. Open
    // with Windows' Unicode API so non-ASCII user names work as well.
#ifdef _WIN32
    FILE * file = _wfopen(std::filesystem::u8path(args[1]).c_str(), L"rb");
#else
    FILE * file = std::fopen(args[1].c_str(), "rb");
#endif
    std::unique_ptr<FILE, decltype(&std::fclose)> model_file(file, std::fclose);
    if (!model_file) {
        reply("Could not open the speech model", 1);
        return 3;
    }
    whisper_model_loader loader{};
    loader.context = model_file.get();
    loader.read = [](void * ctx, void * out, size_t size) {
        return std::fread(out, 1, size, static_cast<FILE *>(ctx));
    };
    loader.eof = [](void * ctx) { return std::feof(static_cast<FILE *>(ctx)) != 0; };
    loader.close = [](void *) {};  // The unique_ptr owns the file.
    std::fprintf(stderr, "worker: loading model\n");
    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
        whisper_init_with_params(&loader, cp), whisper_free);
    model_file.reset();
    if (!ctx || (cp.use_gpu && gpu_failed)) {
        reply("Could not load the speech model", 1);
        return 3;
    }
    // Vulkan compiles pipelines on first use. Pay that cost before READY,
    // rather than letting the first live pass hit its short timeout and
    // invalidate an otherwise working worker. Never publish warmup text.
    auto warm = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    warm.n_threads = threads;
    warm.no_context = warm.no_timestamps = true;
    warm.print_progress = warm.print_realtime = warm.print_timestamps = warm.print_special = false;
    warm.language = "en";
    warm.greedy.best_of = 2;
    std::vector<float> silence(16000, 0.0f);
    if (whisper_full(ctx.get(), warm, silence.data(), int(silence.size()))) {
        reply("Speech worker warmup failed", 1);
        return 3;
    }
    reply(cp.use_gpu ? "READY GPU" : "READY CPU");
    for (;;) {
        std::array<unsigned char, 40> h{};
        if (!read_exact(h.data(), h.size())) return 0;
        if (std::memcmp(h.data(), "WFW1", 4)) return 4;
        uint32_t samples = u32(h.data() + 4), beam = u32(h.data() + 8);
        uint32_t best = u32(h.data() + 12), suppress = u32(h.data() + 16);
        uint32_t context = u32(h.data() + 20), langlen = u32(h.data() + 24);
        uint32_t promptlen = u32(h.data() + 28);
        float temperature = f32(h.data() + 32), threshold = f32(h.data() + 36);
        if (!samples || samples > 16000 * 600 || beam < 1 || beam > 16 ||
            best < 1 || best > 16 || suppress > 1 || context > 1500 ||
            langlen > 32 || promptlen > 65536 || !std::isfinite(temperature) ||
            temperature < 0 || temperature > 1 || !std::isfinite(threshold) ||
            threshold < 0 || threshold > 1) return 4;
        std::string language(langlen, '\0'), prompt(promptlen, '\0');
        std::vector<float> pcm(samples);
        if (!read_exact(language.data(), langlen) || !read_exact(prompt.data(), promptlen) ||
            !read_exact(pcm.data(), pcm.size() * sizeof(float))) return 4;
        if (std::any_of(pcm.begin(), pcm.end(), [](float v) { return !std::isfinite(v); })) return 4;
        auto p = whisper_full_default_params(beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        p.n_threads = threads;
        p.no_context = true;  // Live snapshots are cumulative; never remember the last decode.
        p.no_timestamps = true;
        p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
        p.language = language == "auto" ? nullptr : language.c_str();
        p.detect_language = false;
        p.initial_prompt = prompt.empty() ? nullptr : prompt.c_str();
        p.audio_ctx = int(context);
        p.temperature = temperature;
        p.temperature_inc = 0.2f;
        p.no_speech_thold = threshold;
        p.suppress_nst = suppress != 0;
        p.greedy.best_of = int(best);
        p.beam_search.beam_size = int(beam);
        if (whisper_full(ctx.get(), p, pcm.data(), int(samples))) {
            reply("Speech decode failed", 1);
            continue;
        }
        std::string text;
        for (int i = 0; i < whisper_full_n_segments(ctx.get()); ++i)
            text += whisper_full_get_segment_text(ctx.get(), i);
        reply(text);
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (!size) return 2;
        std::string value(size, '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr);
        value.pop_back();
        args.push_back(value);
    }
#else
int main(int argc, char ** argv) {
    std::vector<std::string> args(argv, argv + argc);
#endif
    try { return run(args); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "worker: %s\n", e.what());
        return 5;
    }
}
