// face_basis — a morphable face model: neutral shape + identity modes +
// expression modes, loaded from the flat binary tools/export_face_basis.py
// writes.
//
// Two bases share one set of coefficients:
//
//   * the render mesh -- what the root scene draws (Maxine/NVF topology by
//     default: 2056 verts, 4048 tris, already cropped to a face mask), and
//   * the landmark basis -- 68 points in dlib order, used only by the fitter.
//
// They are separate because the sources disagree about which they are good at.
// ICT-FaceKit has dlib-68 ordering right, so it supplies the landmarks; Maxine
// has the render topology worth drawing, so it supplies the triangles. The
// exporter resamples ICT's modes onto Maxine's vertices, so evaluating either
// basis with the same (alpha, expression) gives the same face -- there is no
// mapping table to apply here.
//
// Units are ICT's: roughly centimetres, y-up, z toward the viewer, extent about
// +/-7.5 x -10..9.6 x 2.6..13. Nothing here normalises; a consumer that wants a
// unit mesh scales it (RootScene does).

#pragma once

#include <string>
#include <vector>

namespace mirror {

class FaceBasis {
public:
    // Reads the binary. Returns false with `err` set on a missing file, a bad
    // magic, or a truncated one -- a short read is checked explicitly because
    // the arrays are sized by the header, and trusting a corrupt header would
    // read past the buffer rather than fail.
    bool load(const std::string& path, std::string& err);
    bool valid() const { return n_verts_ > 0; }

    int vertexCount() const { return n_verts_; }
    int triangleCount() const { return n_tris_; }
    int identityModes() const { return n_id_; }
    int expressionModes() const { return n_ex_; }
    static constexpr int kLandmarks = 68;
    // True when the render mesh came from Maxine's .nvf rather than ICT.
    bool nvfTopology() const { return nvf_; }

    // Triangle indices into the render mesh, 3 per triangle.
    const std::vector<int>& triangles() const { return tris_; }
    // ARKit-style expression names, in the order the expression modes are
    // stored ("mouthSmile_L", "jawOpen", ...).
    const std::vector<std::string>& expressionNames() const { return ex_names_; }

    // Neutral render mesh, 3 floats per vertex.
    const std::vector<float>& neutral() const { return neutral_; }

    // V = neutral + sum_i alpha_i*id_i + sum_k expr_k*ex_k, into `out`
    // (n_verts*3). Extra coefficients past what the basis carries are ignored,
    // so a caller fitting 80 of 100 identity modes can pass either length.
    void reconstruct(const std::vector<float>& alpha, const std::vector<float>& expr,
                     std::vector<float>& out) const;

    // The same evaluation restricted to the 68 landmark points (68*3). This is
    // what the fitter iterates on -- reconstructing 2056 verts to compare 68
    // points would be ~30x the work for the same answer.
    void reconstructLandmarks(const std::vector<float>& alpha,
                              const std::vector<float>& expr,
                              std::vector<float>& out) const;

    // Raw landmark basis, for the fitter's normal equations.
    // lmNeutral: 68*3. lmIdentity/lmExpression: mode-major, [mode][pt*3+c].
    const std::vector<float>& lmNeutral() const { return lm_neutral_; }
    const std::vector<float>& lmIdentity() const { return lm_id_; }
    const std::vector<float>& lmExpression() const { return lm_ex_; }

private:
    int n_verts_ = 0, n_tris_ = 0, n_id_ = 0, n_ex_ = 0;
    bool nvf_ = false;
    std::vector<float> neutral_, id_, ex_;
    std::vector<float> lm_neutral_, lm_id_, lm_ex_;
    std::vector<int> tris_;
    std::vector<std::string> ex_names_;
};

}  // namespace mirror
