#include "face_basis.h"

#include <cstdio>
#include <cstring>

namespace mirror {

namespace {

// Every read is checked. The header declares the array sizes, so a truncated or
// corrupt file would otherwise have us allocate and read past what is there.
struct Reader {
    std::FILE* f = nullptr;
    bool ok = true;

    template <typename T>
    bool read(T* dst, size_t count) {
        if (!ok) return false;
        ok = std::fread(dst, sizeof(T), count, f) == count;
        return ok;
    }
    bool readVec(std::vector<float>& v, size_t count) {
        v.resize(count);
        return count == 0 || read(v.data(), count);
    }
};

void accumulate(const std::vector<float>& modes, const std::vector<float>& coeff,
                size_t n_modes, size_t stride, std::vector<float>& out) {
    const size_t use = coeff.size() < n_modes ? coeff.size() : n_modes;
    for (size_t m = 0; m < use; ++m) {
        const float c = coeff[m];
        if (c == 0.0f) continue;      // expression vectors are mostly zero
        const float* src = &modes[m * stride];
        for (size_t i = 0; i < stride; ++i) out[i] += c * src[i];
    }
}

}  // namespace

bool FaceBasis::load(const std::string& path, std::string& err) {
    err.clear();
    Reader r;
    r.f = std::fopen(path.c_str(), "rb");
    if (!r.f) {
        err = "face_basis: cannot open " + path +
              " (generate it with tools/export_face_basis.py)";
        return false;
    }

    char magic[4] = {0};
    int hdr[8] = {0};
    r.read(magic, 4);
    r.read(hdr, 8);
    if (!r.ok || std::memcmp(magic, "FBAS", 4) != 0) {
        std::fclose(r.f);
        err = "face_basis: " + path + " is not a face-basis file";
        return false;
    }
    if (hdr[0] != 1) {
        std::fclose(r.f);
        err = "face_basis: unsupported version " + std::to_string(hdr[0]);
        return false;
    }
    n_verts_ = hdr[1];
    n_tris_  = hdr[2];
    n_id_    = hdr[3];
    n_ex_    = hdr[4];
    const int n_lm = hdr[5];
    nvf_ = hdr[6] != 0;
    if (n_verts_ <= 0 || n_tris_ < 0 || n_id_ < 0 || n_ex_ < 0 || n_lm != kLandmarks) {
        std::fclose(r.f);
        err = "face_basis: implausible header in " + path;
        n_verts_ = 0;
        return false;
    }

    const size_t nv3 = size_t(n_verts_) * 3, lm3 = size_t(kLandmarks) * 3;
    r.readVec(neutral_, nv3);
    tris_.resize(size_t(n_tris_) * 3);
    r.read(tris_.data(), tris_.size());
    r.readVec(id_, size_t(n_id_) * nv3);
    r.readVec(ex_, size_t(n_ex_) * nv3);
    r.readVec(lm_neutral_, lm3);
    r.readVec(lm_id_, size_t(n_id_) * lm3);
    r.readVec(lm_ex_, size_t(n_ex_) * lm3);

    ex_names_.clear();
    for (int i = 0; i < n_ex_ && r.ok; ++i) {
        int len = 0;
        if (!r.read(&len, 1)) break;
        if (len < 0 || len > 256) { r.ok = false; break; }
        std::string s(size_t(len), '\0');
        if (len && !r.read(&s[0], size_t(len))) break;
        ex_names_.push_back(std::move(s));
    }

    const bool ok = r.ok && int(ex_names_.size()) == n_ex_;
    std::fclose(r.f);
    if (!ok) {
        err = "face_basis: " + path + " is truncated";
        n_verts_ = 0;
        return false;
    }

    // Triangle indices come from the exporter, but a stray index would index
    // out of the vertex buffer on the GPU, which is not a debuggable failure.
    for (int i : tris_) {
        if (i < 0 || i >= n_verts_) {
            err = "face_basis: triangle index out of range in " + path;
            n_verts_ = 0;
            return false;
        }
    }
    return true;
}

void FaceBasis::reconstruct(const std::vector<float>& alpha,
                            const std::vector<float>& expr,
                            std::vector<float>& out) const {
    const size_t nv3 = size_t(n_verts_) * 3;
    out.assign(neutral_.begin(), neutral_.end());
    if (out.size() != nv3) return;
    accumulate(id_, alpha, size_t(n_id_), nv3, out);
    accumulate(ex_, expr, size_t(n_ex_), nv3, out);
}

void FaceBasis::reconstructLandmarks(const std::vector<float>& alpha,
                                     const std::vector<float>& expr,
                                     std::vector<float>& out) const {
    const size_t lm3 = size_t(kLandmarks) * 3;
    out.assign(lm_neutral_.begin(), lm_neutral_.end());
    if (out.size() != lm3) return;
    accumulate(lm_id_, alpha, size_t(n_id_), lm3, out);
    accumulate(lm_ex_, expr, size_t(n_ex_), lm3, out);
}

}  // namespace mirror
