/*******************************************************************************
Modal Resonator -- Wwise effect plug-in wrapping Mutable Instruments Rings.
*******************************************************************************/

#ifndef ModalResonatorFX_H
#define ModalResonatorFX_H

#include "ModalResonatorFXParams.h"

#include <AK/Plugin/PluginServices/AkFXTailHandler.h>

#include "../../mi_common/mi_block_adapter.h"

#include "rings/dsp/part.h"
#include "rings/dsp/patch.h"
#include "rings/dsp/performance_state.h"
#include "rings/dsp/strummer.h"

/// Rings as a bus/actor-mixer effect: audio on the input excites a modal or
/// string resonator, and onsets in that audio strum it.
class ModalResonatorFX
    : public AK::IAkInPlaceEffectPlugin
{
public:
    ModalResonatorFX();
    ~ModalResonatorFX();

    /// Plug-in initialization, called at the beginning of the plug-in lifetime.
    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    /// Release the resources upon termination of the plug-in.
    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    /// The reinitialization action should be applied here.
    AKRESULT Reset() override;

    /// Plug-in information query mechanism used when the sound engine requires information about the plug-in to determine its behavior.
    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    /// Effect plug-in DSP execution.
    void Execute(AkAudioBuffer* io_pBuffer) override;

    /// Skips execution of some frames, when the voice is virtual playing from elapsed time.
    AKRESULT TimeSkip(AkUInt32 in_uFrames) override;

private:
    /// Pushes the current parameter values into the Rings patch/performance
    /// state. Called once per rendered block, not per sample.
    void UpdatePatch();

    ModalResonatorFXParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkEffectPluginContext* m_pContext;

    AkFXTailHandler m_FXTailHandler;
    AkUInt32 m_uSampleRate;

    /// Rings renders in blocks of at most kMaxBlockSize (24) frames.
    static const size_t kBlockSize = rings::kMaxBlockSize;
    mi::BlockAdapter<kBlockSize, 2> m_adapter;

    rings::Part m_part;
    rings::Strummer m_strummer;
    rings::Patch m_patch;
    rings::PerformanceState m_performanceState;

    /// 32768 uint16_t words, the exact reservation the Rings reverb's FxEngine
    /// makes. Owned by Wwise's allocator rather than being a static array.
    uint16_t* m_pReverbBuffer;
    static const size_t kReverbBufferWords = 32768;

    /// Rings' DSP is tuned for a 48 kHz host. At other rates we correct the
    /// perceived pitch; see the note in Init().
    AkReal32 m_fPitchCorrection;
};

#endif // ModalResonatorFX_H
