/*******************************************************************************
Macro Oscillator -- Wwise source plug-in wrapping Mutable Instruments Plaits.

Plaits is MIT licensed, Copyright 2016 Emilie Gillet. Its sources are compiled
unmodified out of the eurorack repository; this file is the Wwise adaptation.
*******************************************************************************/

#include "MacroOscillatorSource.h"
#include "../MacroOscillatorConfig.h"
#include "../../mi_common/mi_denormal_guard.h"

#include <AK/AkWwiseSDKVersion.h>

#include <cmath>

AK::IAkPlugin* CreateMacroOscillatorSource(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, MacroOscillatorSource());
}

AK::IAkPluginParam* CreateMacroOscillatorSourceParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, MacroOscillatorSourceParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(MacroOscillatorSource, AkPluginTypeSource, MacroOscillatorConfig::CompanyID, MacroOscillatorConfig::PluginID)

namespace
{
    // Plaits emits 16-bit samples in its Frame struct.
    const AkReal32 kShortToFloat = 1.0f / 32768.0f;

    inline AkReal32 dBToLin(AkReal32 in_dB)
    {
        return powf(10.0f, in_dB * 0.05f);
    }
}

MacroOscillatorSource::MacroOscillatorSource()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_uSampleRate(48000)
    , m_uFramePos(plaits::kBlockSize)
    , m_uBlocksRendered(0)
    , m_fPitchCorrection(0.0f)
{
    memset(m_frames, 0, sizeof(m_frames));
}

MacroOscillatorSource::~MacroOscillatorSource()
{
}

AKRESULT MacroOscillatorSource::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkSourcePluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (MacroOscillatorSourceParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;
    m_uSampleRate = in_rFormat.uSampleRate;

    // Plaits is written for 48 kHz; correct the note so the pitch is right at
    // other rates (see the equivalent note in the Rings plug-in).
    m_fPitchCorrection = 12.0f * log2f((AkReal32)m_uSampleRate / plaits::kSampleRate);

    AKRESULT eResult = m_arena.Init(in_pAllocator, kArenaBytes);
    if (eResult != AK_Success)
    {
        return eResult;
    }

    memset(&m_patch, 0, sizeof(m_patch));
    memset(&m_modulations, 0, sizeof(m_modulations));

    // Mono source.
    in_rFormat.channelConfig.SetStandard(AK_SPEAKER_SETUP_MONO);

    m_durationHandler.Setup(m_pParams->NonRTPC.fDuration, 0, m_uSampleRate);

    return Reset();
}

AKRESULT MacroOscillatorSource::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    m_arena.Term(in_pAllocator);
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT MacroOscillatorSource::Reset()
{
    // plaits::Voice has no Reset(): the way to return it to a known state is to
    // rewind its arena and construct it again, which is what the module does on
    // boot. Init() is cheap -- it only lays out pointers and clears state.
    if (m_arena.data())
    {
        m_arena.Clear();
        stmlib::BufferAllocator allocator(m_arena.data(), m_arena.size());
        m_voice.Init(&allocator);
    }

    m_uFramePos = plaits::kBlockSize;
    m_uBlocksRendered = 0;
    memset(m_frames, 0, sizeof(m_frames));
    return AK_Success;
}

AKRESULT MacroOscillatorSource::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeSource;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

AkReal32 MacroOscillatorSource::GetDuration() const
{
    return m_pParams->NonRTPC.fDuration * 1000.0f;
}

void MacroOscillatorSource::UpdatePatch()
{
    const MacroOscillatorRTPCParams& rtpc = m_pParams->RTPC;
    const MacroOscillatorNonRTPCParams& nonRtpc = m_pParams->NonRTPC;

    // Extra parens on AkClamp: the macro's expansion has no outer parens.
    m_patch.note = rtpc.fPitch + m_fPitchCorrection;
    m_patch.harmonics = (AkClamp(rtpc.fHarmonics, 0.0f, 1.0f));
    m_patch.timbre = (AkClamp(rtpc.fTimbre, 0.0f, 1.0f));
    m_patch.morph = (AkClamp(rtpc.fMorph, 0.0f, 1.0f));
    m_patch.decay = (AkClamp(rtpc.fDecay, 0.0f, 1.0f));
    m_patch.lpg_colour = (AkClamp(rtpc.fLpgColour, 0.0f, 1.0f));
    m_patch.frequency_modulation_amount = 0.0f;
    m_patch.timbre_modulation_amount = 0.0f;
    m_patch.morph_modulation_amount = 0.0f;
    m_patch.engine = (AkClamp(nonRtpc.iEngine, 0, plaits::kMaxEngines - 1));

    m_modulations.engine = 0.0f;
    m_modulations.note = 0.0f;
    m_modulations.frequency = 0.0f;
    m_modulations.harmonics = 0.0f;
    m_modulations.timbre = 0.0f;
    m_modulations.morph = 0.0f;
    m_modulations.level = 0.0f;

    m_modulations.frequency_patched = false;
    m_modulations.timbre_patched = false;
    m_modulations.morph_patched = false;
    m_modulations.level_patched = false;

    // With trigger_patched set, Plaits runs its low-pass gate envelope and the
    // voice is percussive. Without it the engine drones continuously, which is
    // what you want for a sustained texture.
    m_modulations.trigger_patched = nonRtpc.bTriggered;
    m_modulations.trigger = 0.0f;
}

void MacroOscillatorSource::Execute(AkAudioBuffer* io_pBuffer)
{
    // See mi_common/mi_denormal_guard.h. The LPG/decay engines and several
    // Plaits engines' internal filters decay toward zero with no denormal
    // protection of their own.
    mi::EnableFlushToZero();

    m_durationHandler.ProduceBuffer(io_pBuffer);

    const AkUInt32 uNumChannels = io_pBuffer->NumChannels();
    const AkUInt16 uFramesToProduce = io_pBuffer->uValidFrames;
    if (uNumChannels == 0 || uFramesToProduce == 0)
    {
        return;
    }

    UpdatePatch();

    const AkReal32 fGain = dBToLin(m_pParams->RTPC.fLevel);
    const AkReal32 fAuxMix = (AkClamp(m_pParams->RTPC.fAuxMix, 0.0f, 1.0f));

    AkReal32* AK_RESTRICT pOut = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(0);

    for (AkUInt16 f = 0; f < uFramesToProduce; ++f)
    {
        if (m_uFramePos >= plaits::kBlockSize)
        {
            // The trigger is a gate, not a pulse. Percussive engines only care
            // about its rising edge, but the three six-operator FM engines read
            // its level as a note-on/note-off gate -- pulsing it for one block
            // releases the DX7 envelope immediately and they render silence.
            // Holding it high for the life of the voice satisfies both.
            //
            // It stays low for the first block so the edge lands after the
            // engine has been loaded and reset.
            if (m_uBlocksRendered > 0)
            {
                m_modulations.trigger = m_pParams->NonRTPC.bTriggered ? 1.0f : 0.0f;
            }
            ++m_uBlocksRendered;

            m_voice.Render(m_patch, m_modulations, m_frames, plaits::kBlockSize);
            m_uFramePos = 0;
        }

        const plaits::Voice::Frame& frame = m_frames[m_uFramePos++];
        const AkReal32 fMain = (AkReal32)frame.out * kShortToFloat;
        const AkReal32 fAux = (AkReal32)frame.aux * kShortToFloat;
        pOut[f] = (fMain + fAuxMix * (fAux - fMain)) * fGain;
    }

    // Mono render, duplicated if the bus asked for more channels.
    for (AkUInt32 c = 1; c < uNumChannels; ++c)
    {
        memcpy(io_pBuffer->GetChannel(c), pOut, uFramesToProduce * sizeof(AkReal32));
    }
}
