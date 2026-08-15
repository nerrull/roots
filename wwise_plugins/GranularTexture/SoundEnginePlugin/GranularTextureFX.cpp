/*******************************************************************************
Granular Texture -- Wwise effect plug-in wrapping Mutable Instruments Clouds.

Clouds is MIT licensed, Copyright 2014 Emilie Gillet. Its sources are compiled
unmodified out of the eurorack repository; this file is the Wwise adaptation.
*******************************************************************************/

#include "GranularTextureFX.h"
#include "../GranularTextureConfig.h"
#include "../../mi_common/mi_denormal_guard.h"

#include <AK/AkWwiseSDKVersion.h>

#include <cmath>

AK::IAkPlugin* CreateGranularTextureFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, GranularTextureFX());
}

AK::IAkPluginParam* CreateGranularTextureFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, GranularTextureFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(GranularTextureFX, AkPluginTypeEffect, GranularTextureConfig::CompanyID, GranularTextureConfig::PluginID)

namespace
{
    // Clouds' buffer holds several seconds of audio and its reverb and feedback
    // paths keep going well after the input stops.
    const AkReal32 kMaxTailSeconds = 10.0f;

    inline short FloatToShort(float in_f)
    {
        float f = in_f * 32768.0f;
        if (f > 32767.0f) f = 32767.0f;
        if (f < -32768.0f) f = -32768.0f;
        return (short)f;
    }
}

GranularTextureFX::GranularTextureFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_uSampleRate(48000)
    , m_uInBlockPos(0)
    , m_uFifoRead(0)
    , m_uFifoWrite(0)
    , m_bRateSupported(true)
{
    memset(m_inBlock, 0, sizeof(m_inBlock));
    memset(m_outBlock, 0, sizeof(m_outBlock));
    memset(m_outFifo, 0, sizeof(m_outFifo));
}

GranularTextureFX::~GranularTextureFX()
{
}

AKRESULT GranularTextureFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (GranularTextureFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;
    m_uSampleRate = in_rFormat.uSampleRate;

    // The 2/3 polyphase conversion is exact only from 48 kHz. At any other host
    // rate we would need a different L/M pair, so rather than silently
    // detuning Clouds' buffer times we run dry.
    m_bRateSupported = (m_uSampleRate == 48000);

    AKRESULT eResult = m_largeArena.Init(in_pAllocator, kLargeBufferBytes);
    if (eResult != AK_Success)
    {
        return eResult;
    }
    eResult = m_smallArena.Init(in_pAllocator, kSmallBufferBytes);
    if (eResult != AK_Success)
    {
        return eResult;
    }

    // On the module GranularProcessor is a global and therefore starts life in
    // zeroed BSS. Some of its state -- notably the WSOLA player's window and
    // search positions -- is only assigned while playing, yet Prepare() reads
    // it before the first Process(). As a plug-in member it is heap memory, so
    // without this the stretch mode's correlator searches on garbage bounds and
    // spins through billions of candidates. It is a plain aggregate with no
    // virtuals, so clearing it is safe.
    memset((void*)&m_processor, 0, sizeof(m_processor));

    m_processor.Init(
        m_largeArena.data(), m_largeArena.size(),
        m_smallArena.data(), m_smallArena.size());

    return Reset();
}

AKRESULT GranularTextureFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    m_largeArena.Term(in_pAllocator);
    m_smallArena.Term(in_pAllocator);
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT GranularTextureFX::Reset()
{
    m_uInBlockPos = 0;
    memset(m_inBlock, 0, sizeof(m_inBlock));
    memset(m_outBlock, 0, sizeof(m_outBlock));
    memset(m_outFifo, 0, sizeof(m_outFifo));

    for (int c = 0; c < 2; ++c)
    {
        m_down[c].Reset();
        m_up[c].Reset();
    }

    // Prime the FIFO with one burst of silence. Output only appears after a
    // whole 32-frame block has been processed, so without this head start the
    // first ~48 host frames would underrun.
    m_uFifoRead = 0;
    m_uFifoWrite = 48;

    return AK_Success;
}

AKRESULT GranularTextureFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void GranularTextureFX::UpdateParameters()
{
    const GranularTextureRTPCParams& rtpc = m_pParams->RTPC;
    const GranularTextureNonRTPCParams& nonRtpc = m_pParams->NonRTPC;

    m_processor.set_playback_mode(
        (clouds::PlaybackMode)(AkClamp(nonRtpc.iPlaybackMode, 0, (AkInt32)clouds::PLAYBACK_MODE_LAST - 1)));
    m_processor.set_quality((AkClamp(nonRtpc.iQuality, 0, 3)));

    clouds::Parameters* p = m_processor.mutable_parameters();

    p->position = (AkClamp(rtpc.fPosition, 0.0f, 1.0f));
    p->size = (AkClamp(rtpc.fSize, 0.0f, 1.0f));
    p->density = (AkClamp(rtpc.fDensity, 0.0f, 1.0f));
    p->texture = (AkClamp(rtpc.fTexture, 0.0f, 1.0f));
    p->stereo_spread = (AkClamp(rtpc.fStereoSpread, 0.0f, 1.0f));
    p->feedback = (AkClamp(rtpc.fFeedback, 0.0f, 1.0f));
    p->reverb = (AkClamp(rtpc.fReverb, 0.0f, 1.0f));

    // Clouds expects pitch in semitones.
    p->pitch = (AkClamp(rtpc.fPitch, -48.0f, 48.0f));

    // The dry/wet crossfade is done here rather than by Clouds so that the
    // plug-in's Dry/Wet behaves like every other Wwise effect. Clouds itself
    // always renders fully wet.
    p->dry_wet = 1.0f;

    p->freeze = rtpc.bFreeze;
    p->trigger = false;
    p->gate = false;
}

void GranularTextureFX::Execute(AkAudioBuffer* io_pBuffer)
{
    // See mi_common/mi_denormal_guard.h -- Clouds' feedback/reverb paths decay
    // toward zero and MI's own code has no denormal protection.
    mi::EnableFlushToZero();

    const AkUInt32 uNumChannels = io_pBuffer->NumChannels();
    const AkUInt16 uValidFrames = io_pBuffer->uValidFrames;
    if (uNumChannels == 0 || !m_bRateSupported)
    {
        return;
    }

    UpdateParameters();

    const AkReal32 fWet = (AkClamp(m_pParams->RTPC.fDryWet, 0.0f, 100.0f)) * 0.01f;
    const AkReal32 fDry = 1.0f - fWet;

    AkReal32* AK_RESTRICT pL = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(0);
    AkReal32* AK_RESTRICT pR = (uNumChannels > 1)
        ? (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(1)
        : pL;

    for (AkUInt16 f = 0; f < uValidFrames; ++f)
    {
        const AkReal32 fInL = pL[f];
        const AkReal32 fInR = pR[f];

        // Host rate -> module rate.
        m_down[0].Push(fInL);
        m_down[1].Push(fInR);

        float l, r;
        while (m_down[0].Pop(&l))
        {
            if (!m_down[1].Pop(&r))
            {
                r = l;
            }

            m_inBlock[m_uInBlockPos].l = FloatToShort(l);
            m_inBlock[m_uInBlockPos].r = FloatToShort(r);

            if (++m_uInBlockPos == kBlockSize)
            {
                m_uInBlockPos = 0;

                // Prepare() handles buffer housekeeping, mode switching and,
                // in stretch mode, advancing the WSOLA correlator search. The
                // firmware calls it continuously from its main loop while
                // Process() runs in the audio interrupt; once per block is the
                // slowest rate that still keeps the correlator ahead of the
                // player.
                m_processor.Prepare();
                m_processor.Process(m_inBlock, m_outBlock, kBlockSize);

                // Module rate -> host rate, into the output FIFO.
                for (size_t i = 0; i < kBlockSize; ++i)
                {
                    m_up[0].Push((float)m_outBlock[i].l / 32768.0f);
                    m_up[1].Push((float)m_outBlock[i].r / 32768.0f);

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

        AkReal32 fWetL = 0.0f;
        AkReal32 fWetR = 0.0f;
        if (FifoCount() > 0)
        {
            fWetL = m_outFifo[0][m_uFifoRead];
            fWetR = m_outFifo[1][m_uFifoRead];
            m_uFifoRead = (m_uFifoRead + 1) & (kOutputFifo - 1);
        }

        pL[f] = fDry * fInL + fWet * fWetL;
        if (uNumChannels > 1)
        {
            pR[f] = fDry * fInR + fWet * fWetR;
        }
    }

    // Any channels beyond the first two get the left channel's result.
    for (AkUInt32 c = 2; c < uNumChannels; ++c)
    {
        memcpy(io_pBuffer->GetChannel(c), pL, uValidFrames * sizeof(AkReal32));
    }

    m_FXTailHandler.HandleTail(io_pBuffer, (AkUInt32)(kMaxTailSeconds * m_uSampleRate));
}

AKRESULT GranularTextureFX::TimeSkip(AkUInt32 in_uFrames)
{
    return AK_DataReady;
}
