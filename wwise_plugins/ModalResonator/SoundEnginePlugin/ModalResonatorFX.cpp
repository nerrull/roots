/*******************************************************************************
Modal Resonator -- Wwise effect plug-in wrapping Mutable Instruments Rings.

Rings is MIT licensed, Copyright 2015 Emilie Gillet. Its sources are compiled
unmodified out of the eurorack repository; this file is the Wwise adaptation.
*******************************************************************************/

#include "ModalResonatorFX.h"
#include "../ModalResonatorConfig.h"

#include <AK/AkWwiseSDKVersion.h>
#include <AK/DSP/AkApplyGain.h>

#include <algorithm>
#include <cmath>

AK::IAkPlugin* CreateModalResonatorFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalResonatorFX());
}

AK::IAkPluginParam* CreateModalResonatorFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalResonatorFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(ModalResonatorFX, AkPluginTypeEffect, ModalResonatorConfig::CompanyID, ModalResonatorConfig::PluginID)

namespace
{
    // Rings' 'damping' knob sets the decay of the resonator. At the top of its
    // range a modal body rings for a very long time, so the tail we report to
    // Wwise has to be generous or notes get cut off when a voice stops.
    const AkReal32 kMaxTailSeconds = 12.0f;

    inline AkReal32 dBToLin(AkReal32 in_dB)
    {
        return powf(10.0f, in_dB * 0.05f);
    }
}

ModalResonatorFX::ModalResonatorFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_uSampleRate(48000)
    , m_pReverbBuffer(nullptr)
    , m_fPitchCorrection(0.0f)
{
}

ModalResonatorFX::~ModalResonatorFX()
{
}

AKRESULT ModalResonatorFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (ModalResonatorFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;
    m_uSampleRate = in_rFormat.uSampleRate;

    // Rings hardcodes a 48 kHz sample rate throughout its DSP (rings/dsp/dsp.h).
    // Running it at another rate transposes everything by the rate ratio, so we
    // pre-compensate the note we hand it. Decay and reverb times still scale
    // with the ratio; at 44.1 kHz that is a ~9% difference, which is audible as
    // slightly longer tails but not as a tuning error.
    m_fPitchCorrection = 12.0f * log2f((AkReal32)m_uSampleRate / rings::kSampleRate);

    m_pReverbBuffer = (uint16_t*)AK_PLUGIN_ALLOC(in_pAllocator, kReverbBufferWords * sizeof(uint16_t));
    if (!m_pReverbBuffer)
    {
        return AK_InsufficientMemory;
    }
    memset(m_pReverbBuffer, 0, kReverbBufferWords * sizeof(uint16_t));

    m_part.Init(m_pReverbBuffer);
    // 0.05 s minimum inter-onset interval: the shortest gap between two strums
    // the onset detector will honour.
    m_strummer.Init(0.01f, rings::kSampleRate / rings::kMaxBlockSize);

    memset(&m_patch, 0, sizeof(m_patch));
    memset(&m_performanceState, 0, sizeof(m_performanceState));

    return Reset();
}

AKRESULT ModalResonatorFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    if (m_pReverbBuffer)
    {
        AK_PLUGIN_FREE(in_pAllocator, m_pReverbBuffer);
        m_pReverbBuffer = nullptr;
    }
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT ModalResonatorFX::Reset()
{
    m_adapter.Reset();
    if (m_pReverbBuffer)
    {
        memset(m_pReverbBuffer, 0, kReverbBufferWords * sizeof(uint16_t));
        m_part.Init(m_pReverbBuffer);
    }
    return AK_Success;
}

AKRESULT ModalResonatorFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void ModalResonatorFX::UpdatePatch()
{
    const ModalResonatorRTPCParams& rtpc = m_pParams->RTPC;
    const ModalResonatorNonRTPCParams& nonRtpc = m_pParams->NonRTPC;

    m_patch.structure = (AkClamp(rtpc.fStructure, 0.0f, 0.9995f));
    m_patch.brightness = (AkClamp(rtpc.fBrightness, 0.0f, 1.0f));
    m_patch.damping = (AkClamp(rtpc.fDamping, 0.0f, 1.0f));
    m_patch.position = (AkClamp(rtpc.fPosition, 0.0f, 1.0f));

    // Rings expects the note as a MIDI note number; 'tonic' is the root the
    // chord table is built from.
    m_performanceState.note = rtpc.fPitch + m_fPitchCorrection;
    m_performanceState.tonic = 12.0f;
    m_performanceState.fm = 0.0f;
    m_performanceState.chord = (AkClamp(nonRtpc.iChord, 0, rings::kNumChords - 1));

    // The input bus is the exciter, and we let the strummer fire the resonator
    // from onsets it detects in that audio. internal_note keeps the pitch under
    // RTPC control rather than letting the strummer track a CV input.
    m_performanceState.internal_exciter = nonRtpc.bInternalExciter;
    m_performanceState.internal_strum = true;
    m_performanceState.internal_note = true;

    // Note the extra parentheses on every AkClamp: it is a macro whose
    // expansion has no outer parens, so anything applied to the result binds
    // to the last branch of its ternary instead of the whole expression.
    m_part.set_model((rings::ResonatorModel)(AkClamp(nonRtpc.iModel, 0, (AkInt32)rings::RESONATOR_MODEL_LAST - 1)));
    m_part.set_polyphony((AkClamp(nonRtpc.iPolyphony, 1, rings::kMaxPolyphony)));
}

void ModalResonatorFX::Execute(AkAudioBuffer* io_pBuffer)
{
    const AkUInt32 uNumChannels = io_pBuffer->NumChannels();
    const AkUInt16 uValidFrames = io_pBuffer->uValidFrames;
    if (uNumChannels == 0)
    {
        return;
    }

    const AkReal32 fWet = (AkClamp(m_pParams->RTPC.fDryWet, 0.0f, 100.0f)) * 0.01f;
    const AkReal32 fDry = 1.0f - fWet;
    const AkReal32 fGain = dBToLin(m_pParams->RTPC.fOutputLevel);
    const AkReal32 fSpread = (AkClamp(m_pParams->RTPC.fSpread, 0.0f, 1.0f));

    UpdatePatch();

    // Rings takes a mono exciter. Sum whatever the bus gives us into one
    // stream, and keep the per-channel dry signal for the mix.
    const AkReal32 fInScale = 1.0f / (AkReal32)uNumChannels;

    auto render = [this](const float* in, float* const* out, size_t n)
    {
        // The strummer inspects the block that is about to be rendered and
        // decides whether this block starts a new note.
        m_performanceState.strum = false;
        m_strummer.Process(in, n, &m_performanceState);
        m_part.Process(m_performanceState, m_patch, in, out[0], out[1], n);
    };

    for (AkUInt16 f = 0; f < uValidFrames; ++f)
    {
        AkReal32 fIn = 0.0f;
        for (AkUInt32 c = 0; c < uNumChannels; ++c)
        {
            fIn += io_pBuffer->GetChannel(c)[f];
        }
        fIn *= fInScale;

        AkReal32 wet[2];
        m_adapter.Tick(fIn, wet, render);

        // 'out' and 'aux' are Rings' two outputs. At spread 0 they are summed
        // to a mono image, at spread 1 they land on separate channels.
        const AkReal32 fMono = 0.5f * (wet[0] + wet[1]);
        const AkReal32 fLeft = fMono + fSpread * (wet[0] - fMono);
        const AkReal32 fRight = fMono + fSpread * (wet[1] - fMono);

        for (AkUInt32 c = 0; c < uNumChannels; ++c)
        {
            AkReal32* AK_RESTRICT pBuf = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(c);
            const AkReal32 fWetCh = (c & 1) ? fRight : fLeft;
            pBuf[f] = (fDry * pBuf[f] + fWet * fWetCh) * fGain;
        }
    }

    // The resonator keeps ringing after the input goes silent, so ask Wwise to
    // keep pulling on us for the length of the tail.
    m_FXTailHandler.HandleTail(io_pBuffer, (AkUInt32)(kMaxTailSeconds * m_uSampleRate));
}

AKRESULT ModalResonatorFX::TimeSkip(AkUInt32 in_uFrames)
{
    return AK_DataReady;
}
