/*******************************************************************************
Modal Voice -- Wwise effect plug-in wrapping Mutable Instruments Elements.

Elements is MIT licensed, Copyright 2014 Emilie Gillet. Its sources are compiled
unmodified out of the eurorack repository; this file is the Wwise adaptation.
*******************************************************************************/

#include "ModalVoiceFX.h"
#include "../ModalVoiceConfig.h"
#include "../../mi_common/mi_denormal_guard.h"

#include <AK/AkWwiseSDKVersion.h>

#include <cmath>

AK::IAkPlugin* CreateModalVoiceFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalVoiceFX());
}

AK::IAkPluginParam* CreateModalVoiceFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalVoiceFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(ModalVoiceFX, AkPluginTypeEffect, ModalVoiceConfig::CompanyID, ModalVoiceConfig::PluginID)

namespace
{
    // Elements' resonator plus its reverb can ring for a long time at high
    // damping settings.
    const AkReal32 kMaxTailSeconds = 12.0f;

    inline AkReal32 dBToLin(AkReal32 in_dB)
    {
        return powf(10.0f, in_dB * 0.05f);
    }
}

ModalVoiceFX::ModalVoiceFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_uSampleRate(48000)
    , m_pReverbBuffer(nullptr)
    , m_uInBlockPos(0)
    , m_uFifoRead(0)
    , m_uFifoWrite(0)
    , m_bRateSupported(true)
{
    memset(m_blowIn, 0, sizeof(m_blowIn));
    memset(m_strikeIn, 0, sizeof(m_strikeIn));
    memset(m_mainOut, 0, sizeof(m_mainOut));
    memset(m_auxOut, 0, sizeof(m_auxOut));
    memset(m_outFifo, 0, sizeof(m_outFifo));
    memset(&m_performanceState, 0, sizeof(m_performanceState));
}

ModalVoiceFX::~ModalVoiceFX()
{
}

AKRESULT ModalVoiceFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (ModalVoiceFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;
    m_uSampleRate = in_rFormat.uSampleRate;

    // The 2/3 conversion around the 32 kHz core is exact only from 48 kHz.
    m_bRateSupported = (m_uSampleRate == 48000);

    m_pReverbBuffer = (uint16_t*)AK_PLUGIN_ALLOC(in_pAllocator, kReverbBufferWords * sizeof(uint16_t));
    if (!m_pReverbBuffer)
    {
        return AK_InsufficientMemory;
    }

    // Elements' Part is a global in the firmware and so starts out zeroed; as a
    // plug-in member it is heap memory. Clear it before Init for the same
    // reason the Clouds plug-in does.
    memset((void*)&m_part, 0, sizeof(m_part));
    memset(m_pReverbBuffer, 0, kReverbBufferWords * sizeof(uint16_t));

    m_part.Init(m_pReverbBuffer);

    // Seeds the per-instance variation in the exciter models. The firmware uses
    // the chip's unique ID; any stable value does the job here.
    uint32_t seed[3] = { 0x9e3779b9, 0x243f6a88, 0xb7e15162 };
    m_part.Seed(seed, 3);

    return Reset();
}

AKRESULT ModalVoiceFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    if (m_pReverbBuffer)
    {
        AK_PLUGIN_FREE(in_pAllocator, m_pReverbBuffer);
        m_pReverbBuffer = nullptr;
    }
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT ModalVoiceFX::Reset()
{
    m_uInBlockPos = 0;
    memset(m_blowIn, 0, sizeof(m_blowIn));
    memset(m_strikeIn, 0, sizeof(m_strikeIn));
    memset(m_mainOut, 0, sizeof(m_mainOut));
    memset(m_auxOut, 0, sizeof(m_auxOut));
    memset(m_outFifo, 0, sizeof(m_outFifo));

    m_down.Reset();
    m_up[0].Reset();
    m_up[1].Reset();
    m_exciterAGC.Reset();

    // One block of output arrives at a time, so prime the FIFO to cover it.
    m_uFifoRead = 0;
    m_uFifoWrite = 24;

    return AK_Success;
}

AKRESULT ModalVoiceFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void ModalVoiceFX::UpdatePatch()
{
    const ModalVoiceRTPCParams& rtpc = m_pParams->RTPC;
    const ModalVoiceNonRTPCParams& nonRtpc = m_pParams->NonRTPC;

    elements::Patch* p = m_part.mutable_patch();

    p->exciter_envelope_shape = (AkClamp(rtpc.fContour, 0.0f, 1.0f));
    p->exciter_bow_level = (AkClamp(rtpc.fBowLevel, 0.0f, 1.0f));
    p->exciter_blow_level = (AkClamp(rtpc.fBlowLevel, 0.0f, 1.0f));
    p->exciter_blow_meta = (AkClamp(rtpc.fBlowMeta, 0.0f, 1.0f));
    p->exciter_strike_level = (AkClamp(rtpc.fStrikeLevel, 0.0f, 1.0f));
    p->exciter_strike_meta = (AkClamp(rtpc.fStrikeMeta, 0.0f, 1.0f));

    // Timbre and signature are left at neutral values rather than exposed;
    // they interact heavily with the meta controls and add little on a bus.
    p->exciter_bow_timbre = 0.5f;
    p->exciter_blow_timbre = 0.5f;
    p->exciter_strike_timbre = 0.5f;
    p->exciter_signature = 0.0f;

    p->resonator_geometry = (AkClamp(rtpc.fGeometry, 0.0f, 1.0f));
    p->resonator_brightness = (AkClamp(rtpc.fBrightness, 0.0f, 1.0f));
    p->resonator_damping = (AkClamp(rtpc.fDamping, 0.0f, 1.0f));
    p->resonator_position = (AkClamp(rtpc.fPosition, 0.0f, 1.0f));
    p->resonator_modulation_frequency = 0.5f;
    p->resonator_modulation_offset = 0.0f;
    p->modulation_frequency = 0.0f;

    p->reverb_diffusion = 0.625f;
    p->reverb_lp = 0.7f;
    p->space = (AkClamp(rtpc.fSpace, 0.0f, 1.0f));

    m_performanceState.note = rtpc.fPitch;
    m_performanceState.modulation = 0.0f;
    m_performanceState.strength = (AkClamp(rtpc.fStrength, 0.0f, 1.0f));
    m_performanceState.gate = nonRtpc.bGate;
}

void ModalVoiceFX::Execute(AkAudioBuffer* io_pBuffer)
{
    // See mi_common/mi_denormal_guard.h -- Elements' resonator and reverb
    // decay toward zero and MI's own code has no denormal protection.
    mi::EnableFlushToZero();

    const AkUInt32 uNumChannels = io_pBuffer->NumChannels();
    const AkUInt16 uValidFrames = io_pBuffer->uValidFrames;
    if (uNumChannels == 0 || !m_bRateSupported)
    {
        return;
    }

    UpdatePatch();

    const AkReal32 fWet = (AkClamp(m_pParams->RTPC.fDryWet, 0.0f, 100.0f)) * 0.01f;
    const AkReal32 fDry = 1.0f - fWet;
    const AkReal32 fGain = dBToLin(m_pParams->RTPC.fLevel);
    const AkReal32 fSpread = (AkClamp(m_pParams->RTPC.fSpread, 0.0f, 1.0f));
    const bool bExternal = m_pParams->NonRTPC.bExternalExciter;

    const AkReal32 fInScale = 1.0f / (AkReal32)uNumChannels;

    AkReal32* AK_RESTRICT pL = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(0);
    AkReal32* AK_RESTRICT pR = (uNumChannels > 1)
        ? (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(1)
        : pL;

    for (AkUInt16 f = 0; f < uValidFrames; ++f)
    {
        AkReal32 fIn = 0.0f;
        for (AkUInt32 c = 0; c < uNumChannels; ++c)
        {
            fIn += io_pBuffer->GetChannel(c)[f];
        }
        fIn *= fInScale;

        // Elements' exciter inputs are fed straight into the resonator with
        // no gain stage of their own, tuned for Eurorack line level. Real bus
        // audio is usually much quieter, which otherwise shows up as a wet
        // signal too faint to hear. Only worth computing when it will
        // actually be used.
        const AkReal32 fExciter = bExternal ? m_exciterAGC.Process(fIn) : fIn;

        m_down.Push(fExciter);

        float x;
        while (m_down.Pop(&x))
        {
            // Elements takes two exciter inputs: a continuous one for the
            // blow/bow models and a transient one for the strike model. Feeding
            // the same signal to both lets a single bus drive either.
            m_blowIn[m_uInBlockPos] = bExternal ? x : 0.0f;
            m_strikeIn[m_uInBlockPos] = bExternal ? x : 0.0f;

            if (++m_uInBlockPos == kBlockSize)
            {
                m_uInBlockPos = 0;
                m_part.Process(
                    m_performanceState,
                    m_blowIn,
                    m_strikeIn,
                    m_mainOut,
                    m_auxOut,
                    kBlockSize);

                for (size_t i = 0; i < kBlockSize; ++i)
                {
                    m_up[0].Push(m_mainOut[i]);
                    m_up[1].Push(m_auxOut[i]);

                    float ol, orr;
                    while (m_up[0].Pop(&ol))
                    {
                        if (!m_up[1].Pop(&orr))
                        {
                            orr = ol;
                        }
                        m_outFifo[0][m_uFifoWrite] = ol;
                        m_outFifo[1][m_uFifoWrite] = orr;
                        m_uFifoWrite = (m_uFifoWrite + 1) & (kOutputFifo - 1);
                    }
                }
            }
        }

        AkReal32 fMain = 0.0f;
        AkReal32 fAux = 0.0f;
        if (FifoCount() > 0)
        {
            fMain = m_outFifo[0][m_uFifoRead];
            fAux = m_outFifo[1][m_uFifoRead];
            m_uFifoRead = (m_uFifoRead + 1) & (kOutputFifo - 1);
        }

        const AkReal32 fMono = 0.5f * (fMain + fAux);
        const AkReal32 fWetL = fMono + fSpread * (fMain - fMono);
        const AkReal32 fWetR = fMono + fSpread * (fAux - fMono);

        pL[f] = (fDry * pL[f] + fWet * fWetL) * fGain;
        if (uNumChannels > 1)
        {
            pR[f] = (fDry * pR[f] + fWet * fWetR) * fGain;
        }
    }

    for (AkUInt32 c = 2; c < uNumChannels; ++c)
    {
        memcpy(io_pBuffer->GetChannel(c), pL, uValidFrames * sizeof(AkReal32));
    }

    m_FXTailHandler.HandleTail(io_pBuffer, (AkUInt32)(kMaxTailSeconds * m_uSampleRate));
}

AKRESULT ModalVoiceFX::TimeSkip(AkUInt32 in_uFrames)
{
    return AK_DataReady;
}
