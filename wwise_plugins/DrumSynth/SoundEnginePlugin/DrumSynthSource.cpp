/*******************************************************************************
Drum Synth -- Wwise source plug-in wrapping Mutable Instruments Peaks.

Peaks is MIT licensed, Copyright 2013 Emilie Gillet.
*******************************************************************************/

#include "DrumSynthSource.h"
#include "../DrumSynthConfig.h"
#include "../../mi_common/mi_denormal_guard.h"

#include <AK/AkWwiseSDKVersion.h>

#include <cmath>

AK::IAkPlugin* CreateDrumSynthSource(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, DrumSynthSource());
}

AK::IAkPluginParam* CreateDrumSynthSourceParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, DrumSynthSourceParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(DrumSynthSource, AkPluginTypeSource, DrumSynthConfig::CompanyID, DrumSynthConfig::PluginID)

namespace
{
    const AkReal32 kShortToFloat = 1.0f / 32768.0f;

    // Peaks renders in chunks; this is just how many samples we ask for at a
    // time so the gate array stays a fixed, stack-friendly size.
    const size_t kChunkSize = 64;

    // The drum models trigger on the rising edge but read the gate for a short
    // accent window, so a 2 ms pulse is what the module's own front panel does.
    const AkReal32 kGateSeconds = 0.002f;

    // Peaks' four knobs are 16-bit unsigned on the module.
    inline AkUInt16 ToPeaksParam(AkReal32 in_fNormalized)
    {
        const AkReal32 f = (AkClamp(in_fNormalized, 0.0f, 1.0f));
        return (AkUInt16)(f * 65535.0f);
    }

    inline AkReal32 dBToLin(AkReal32 in_dB)
    {
        return powf(10.0f, in_dB * 0.05f);
    }

    // Only the percussion models make sense as a Wwise source; the envelope,
    // LFO and sequencer functions are control-rate generators.
    peaks::ProcessorFunction ModelToFunction(AkInt32 in_iModel)
    {
        switch (in_iModel)
        {
        case 1:  return peaks::PROCESSOR_FUNCTION_SNARE_DRUM;
        case 2:  return peaks::PROCESSOR_FUNCTION_HIGH_HAT;
        case 3:  return peaks::PROCESSOR_FUNCTION_FM_DRUM;
        case 0:
        default: return peaks::PROCESSOR_FUNCTION_BASS_DRUM;
        }
    }
}

DrumSynthSource::DrumSynthSource()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_uSampleRate(48000)
    , m_uSamplesRendered(0)
    , m_uGateLengthSamples(96)
    , m_iCurrentModel(-1)
{
    memset(m_auCurrentParams, 0xff, sizeof(m_auCurrentParams));
}

DrumSynthSource::~DrumSynthSource()
{
}

AKRESULT DrumSynthSource::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkSourcePluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (DrumSynthSourceParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;
    m_uSampleRate = in_rFormat.uSampleRate;

    m_uGateLengthSamples = (AkUInt32)(kGateSeconds * m_uSampleRate);

    m_processor.Init(0);
    m_processor.set_control_mode(peaks::CONTROL_MODE_FULL);

    in_rFormat.channelConfig.SetStandard(AK_SPEAKER_SETUP_MONO);
    m_durationHandler.Setup(m_pParams->NonRTPC.fDuration, 0, m_uSampleRate);

    return Reset();
}

AKRESULT DrumSynthSource::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT DrumSynthSource::Reset()
{
    m_uSamplesRendered = 0;
    // Force the next UpdateParameters() to reconfigure from scratch.
    m_iCurrentModel = -1;
    memset(m_auCurrentParams, 0xff, sizeof(m_auCurrentParams));
    return AK_Success;
}

AKRESULT DrumSynthSource::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeSource;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

AkReal32 DrumSynthSource::GetDuration() const
{
    return m_pParams->NonRTPC.fDuration * 1000.0f;
}

void DrumSynthSource::UpdateParameters()
{
    const DrumSynthRTPCParams& rtpc = m_pParams->RTPC;

    const AkInt32 iModel = (AkClamp(m_pParams->NonRTPC.iModel, 0, 3));
    if (iModel != m_iCurrentModel)
    {
        m_processor.set_function(ModelToFunction(iModel));
        m_iCurrentModel = iModel;
    }

    const AkUInt16 auParams[4] =
    {
        ToPeaksParam(rtpc.fParam1),
        ToPeaksParam(rtpc.fParam2),
        ToPeaksParam(rtpc.fParam3),
        ToPeaksParam(rtpc.fParam4),
    };

    for (int i = 0; i < 4; ++i)
    {
        if (auParams[i] != m_auCurrentParams[i])
        {
            m_processor.set_parameter((uint8_t)i, auParams[i]);
            m_auCurrentParams[i] = auParams[i];
        }
    }
}

void DrumSynthSource::Execute(AkAudioBuffer* io_pBuffer)
{
    // See mi_common/mi_denormal_guard.h -- Peaks' envelope/decay tails have no
    // denormal protection of their own.
    mi::EnableFlushToZero();

    m_durationHandler.ProduceBuffer(io_pBuffer);

    const AkUInt32 uNumChannels = io_pBuffer->NumChannels();
    const AkUInt16 uFramesToProduce = io_pBuffer->uValidFrames;
    if (uNumChannels == 0 || uFramesToProduce == 0)
    {
        return;
    }

    UpdateParameters();

    const AkReal32 fGain = dBToLin(m_pParams->RTPC.fLevel);
    AkReal32* AK_RESTRICT pOut = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(0);

    stmlib::GateFlags gateFlags[kChunkSize];
    int16_t samples[kChunkSize];

    AkUInt16 uProduced = 0;
    while (uProduced < uFramesToProduce)
    {
        const size_t uChunk = AkMin(kChunkSize, (size_t)(uFramesToProduce - uProduced));

        for (size_t i = 0; i < uChunk; ++i)
        {
            const AkUInt32 uSample = m_uSamplesRendered + (AkUInt32)i;
            if (uSample == 0)
            {
                gateFlags[i] = stmlib::GATE_FLAG_RISING | stmlib::GATE_FLAG_HIGH;
            }
            else if (uSample < m_uGateLengthSamples)
            {
                gateFlags[i] = stmlib::GATE_FLAG_HIGH;
            }
            else if (uSample == m_uGateLengthSamples)
            {
                gateFlags[i] = stmlib::GATE_FLAG_FALLING;
            }
            else
            {
                gateFlags[i] = stmlib::GATE_FLAG_LOW;
            }
        }

        m_processor.Process(gateFlags, samples, uChunk);

        for (size_t i = 0; i < uChunk; ++i)
        {
            pOut[uProduced + i] = (AkReal32)samples[i] * kShortToFloat * fGain;
        }

        m_uSamplesRendered += (AkUInt32)uChunk;
        uProduced += (AkUInt16)uChunk;
    }

    for (AkUInt32 c = 1; c < uNumChannels; ++c)
    {
        memcpy(io_pBuffer->GetChannel(c), pOut, uFramesToProduce * sizeof(AkReal32));
    }
}
