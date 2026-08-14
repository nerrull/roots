/*******************************************************************************
Drum Synth -- Wwise source plug-in wrapping Mutable Instruments Peaks.
*******************************************************************************/

#ifndef DrumSynthSource_H
#define DrumSynthSource_H

#include "DrumSynthSourceParams.h"

#include <AK/Plugin/PluginServices/AkFXDurationHandler.h>

#include "stmlib/utils/gate_flags.h"
#include "peaks/processors.h"

class DrumSynthSource
    : public AK::IAkSourcePlugin
{
public:
    DrumSynthSource();
    ~DrumSynthSource();

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkSourcePluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Reset() override;

    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    void Execute(AkAudioBuffer* io_pBuffer) override;

    AkReal32 GetDuration() const override;

private:
    void UpdateParameters();

    DrumSynthSourceParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkSourcePluginContext* m_pContext;
    AkFXDurationHandler m_durationHandler;

    AkUInt32 m_uSampleRate;

    peaks::Processors m_processor;

    /// Peaks is driven by a gate signal. The voice starting is the rising edge;
    /// we hold the gate high briefly so the drum models see a real trigger
    /// pulse rather than a single-sample glitch.
    AkUInt32 m_uSamplesRendered;
    AkUInt32 m_uGateLengthSamples;

    /// Cached so we only reconfigure the processor when something changed --
    /// Configure() re-derives every model coefficient.
    AkInt32 m_iCurrentModel;
    AkUInt16 m_auCurrentParams[4];
};

#endif // DrumSynthSource_H
