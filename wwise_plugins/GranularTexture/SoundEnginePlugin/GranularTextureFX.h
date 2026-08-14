/*******************************************************************************
Granular Texture -- Wwise effect plug-in wrapping Mutable Instruments Clouds.
*******************************************************************************/

#ifndef GranularTextureFX_H
#define GranularTextureFX_H

#include "GranularTextureFXParams.h"

#include <AK/Plugin/PluginServices/AkFXTailHandler.h>

#include "../../mi_common/mi_arena.h"
#include "../../mi_common/mi_resampler.h"

#include "clouds/dsp/granular_processor.h"
#include "clouds/dsp/frame.h"

/// Clouds as a bus effect. The signal path per channel is:
///
///   host 48 kHz -> downsample 2/3 -> 32 kHz block of 32 -> GranularProcessor
///                -> upsample 3/2 -> host 48 kHz
///
/// The block and the rate conversion each add a little latency; see kOutputFifo.
class GranularTextureFX
    : public AK::IAkInPlaceEffectPlugin
{
public:
    GranularTextureFX();
    ~GranularTextureFX();

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Reset() override;

    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    void Execute(AkAudioBuffer* io_pBuffer) override;

    AKRESULT TimeSkip(AkUInt32 in_uFrames) override;

private:
    void UpdateParameters();

    GranularTextureFXParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkEffectPluginContext* m_pContext;

    AkFXTailHandler m_FXTailHandler;
    AkUInt32 m_uSampleRate;

    /// Clouds' two working buffers. On the module these are external SDRAM and
    /// internal SRAM; the sizes are the ones its own firmware uses.
    mi::Arena m_largeArena;
    mi::Arena m_smallArena;
    static const size_t kLargeBufferBytes = 118784;
    static const size_t kSmallBufferBytes = 65536 - 128;

    clouds::GranularProcessor m_processor;

    /// Clouds renders 32 frames at a time at 32 kHz.
    static const size_t kBlockSize = 32;
    clouds::ShortFrame m_inBlock[kBlockSize];
    clouds::ShortFrame m_outBlock[kBlockSize];
    size_t m_uInBlockPos;

    mi::HostToModule48to32 m_down[2];
    mi::ModuleToHost32to48 m_up[2];

    /// Output arrives in bursts of 48 host frames after every 48 host frames of
    /// input, so the FIFO has to hold at least one burst to avoid underrunning.
    static const size_t kOutputFifo = 256;
    float m_outFifo[2][kOutputFifo];
    size_t m_uFifoRead;
    size_t m_uFifoWrite;

    inline size_t FifoCount() const
    {
        return (m_uFifoWrite - m_uFifoRead) & (kOutputFifo - 1);
    }

    /// True once the host sample rate has been checked; Clouds only gets its
    /// exact 2/3 conversion at 48 kHz.
    bool m_bRateSupported;
};

#endif // GranularTextureFX_H
