/*******************************************************************************
Macro Oscillator -- Wwise source plug-in wrapping Mutable Instruments Plaits.
*******************************************************************************/

#ifndef MacroOscillatorSource_H
#define MacroOscillatorSource_H

#include "MacroOscillatorSourceParams.h"

#include <AK/Plugin/PluginServices/AkFXDurationHandler.h>

#include "../../mi_common/mi_arena.h"

#include "stmlib/utils/buffer_allocator.h"
#include "plaits/dsp/voice.h"

class MacroOscillatorSource
    : public AK::IAkSourcePlugin
{
public:
    MacroOscillatorSource();
    ~MacroOscillatorSource();

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkSourcePluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Reset() override;

    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    void Execute(AkAudioBuffer* io_pBuffer) override;

    AkReal32 GetDuration() const override;

private:
    void UpdatePatch();

    MacroOscillatorSourceParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkSourcePluginContext* m_pContext;
    AkFXDurationHandler m_durationHandler;

    AkUInt32 m_uSampleRate;

    /// Plaits sub-allocates all of its engine state out of one 16 KB block,
    /// exactly as it does from SRAM on the module.
    mi::Arena m_arena;
    static const size_t kArenaBytes = 16384;

    plaits::Voice m_voice;
    plaits::Patch m_patch;
    plaits::Modulations m_modulations;

    /// Plaits renders in blocks of kBlockSize (12) frames; we accumulate its
    /// output here and drain it across Wwise's buffers.
    plaits::Voice::Frame m_frames[plaits::kBlockSize];
    size_t m_uFramePos;

    /// Counts rendered blocks since Reset; used only to keep the trigger gate
    /// low for the first block so its rising edge lands after engine load.
    AkUInt32 m_uBlocksRendered;

    AkReal32 m_fPitchCorrection;
};

#endif // MacroOscillatorSource_H
