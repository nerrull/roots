/*******************************************************************************
Modal Voice -- Wwise effect plug-in wrapping Mutable Instruments Elements.
*******************************************************************************/

#ifndef ModalVoiceFX_H
#define ModalVoiceFX_H

#include "ModalVoiceFXParams.h"

#include <AK/Plugin/PluginServices/AkFXTailHandler.h>

#include "../../mi_common/mi_resampler.h"

#include "elements/dsp/part.h"
#include "elements/dsp/patch.h"

/// Elements as a bus effect. Like the module's external-input mode, the audio
/// on the bus is fed to the exciter inputs and the modal resonator is what you
/// hear. Elements runs at 32 kHz, so the signal is rate-converted around it.
class ModalVoiceFX
    : public AK::IAkInPlaceEffectPlugin
{
public:
    ModalVoiceFX();
    ~ModalVoiceFX();

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Reset() override;

    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    void Execute(AkAudioBuffer* io_pBuffer) override;

    AKRESULT TimeSkip(AkUInt32 in_uFrames) override;

private:
    void UpdatePatch();

    ModalVoiceFXParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkEffectPluginContext* m_pContext;

    AkFXTailHandler m_FXTailHandler;
    AkUInt32 m_uSampleRate;

    elements::Part m_part;
    elements::PerformanceState m_performanceState;

    /// The exact reservation made by the Elements reverb's FxEngine.
    uint16_t* m_pReverbBuffer;
    static const size_t kReverbBufferWords = 32768;

    /// Elements renders at most kMaxBlockSize (16) frames per call.
    static const size_t kBlockSize = elements::kMaxBlockSize;
    float m_blowIn[kBlockSize];
    float m_strikeIn[kBlockSize];
    float m_mainOut[kBlockSize];
    float m_auxOut[kBlockSize];
    size_t m_uInBlockPos;

    mi::HostToModule48to32 m_down;
    mi::ModuleToHost32to48 m_up[2];

    static const size_t kOutputFifo = 256;
    float m_outFifo[2][kOutputFifo];
    size_t m_uFifoRead;
    size_t m_uFifoWrite;

    inline size_t FifoCount() const
    {
        return (m_uFifoWrite - m_uFifoRead) & (kOutputFifo - 1);
    }

    bool m_bRateSupported;
};

#endif // ModalVoiceFX_H
