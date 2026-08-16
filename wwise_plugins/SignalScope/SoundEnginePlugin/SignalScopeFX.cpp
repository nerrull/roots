/*******************************************************************************
The content of this file includes portions of the AUDIOKINETIC Wwise Technology
released in source code form as part of the SDK installer package.

Commercial License Usage

Licensees holding valid commercial licenses to the AUDIOKINETIC Wwise Technology
may use this file in accordance with the end user license agreement provided
with the software or, alternatively, in accordance with the terms contained in a
written agreement between you and Audiokinetic Inc.

Apache License Usage

Alternatively, this file may be used under the Apache License, Version 2.0 (the
"Apache License"); you may not use this file except in compliance with the
Apache License. You may obtain a copy of the Apache License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the Apache License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License for
the specific language governing permissions and limitations under the License.

  Copyright (c) 2026 Audiokinetic Inc.
*******************************************************************************/

#include "SignalScopeFX.h"
#include "../SignalScopeConfig.h"

#include <AK/AkWwiseSDKVersion.h>
#include <AK/SoundEngine/Common/AkSpeakerConfig.h>

#include <cstdio>
#include <cstring>

namespace
{
    // Speaker-bit -> short label, in AK_SPEAKER_* bit order (not channel
    // index order -- AK::ChannelBitToIndex below handles that, including its
    // "LFE always goes last" rule).
    struct SpeakerName { AkUInt32 bit; const char* name; };
    const SpeakerName kSpeakerNames[] = {
        { AK_SPEAKER_FRONT_LEFT,          "FL"  },
        { AK_SPEAKER_FRONT_RIGHT,         "FR"  },
        { AK_SPEAKER_FRONT_CENTER,        "FC"  },
        { AK_SPEAKER_LOW_FREQUENCY,       "LFE" },
        { AK_SPEAKER_BACK_LEFT,           "BL"  },
        { AK_SPEAKER_BACK_RIGHT,          "BR"  },
        { AK_SPEAKER_BACK_CENTER,         "BC"  },
        { AK_SPEAKER_SIDE_LEFT,           "SL"  },
        { AK_SPEAKER_SIDE_RIGHT,          "SR"  },
        { AK_SPEAKER_TOP,                 "TC"  },
        { AK_SPEAKER_HEIGHT_FRONT_LEFT,   "TFL" },
        { AK_SPEAKER_HEIGHT_FRONT_CENTER, "TFC" },
        { AK_SPEAKER_HEIGHT_FRONT_RIGHT,  "TFR" },
        { AK_SPEAKER_HEIGHT_BACK_LEFT,    "TBL" },
        { AK_SPEAKER_HEIGHT_BACK_CENTER,  "TBC" },
        { AK_SPEAKER_HEIGHT_BACK_RIGHT,   "TBR" },
        { AK_SPEAKER_HEIGHT_SIDE_LEFT,    "TSL" },
        { AK_SPEAKER_HEIGHT_SIDE_RIGHT,   "TSR" },
    };

    // Fills names[0..numChannels) with short speaker labels for a standard
    // channel config (e.g. "FL"/"FR"/"LFE" for 2.1), matching the same index
    // order AkAudioBuffer::GetChannel(i) uses. Anonymous/ambisonic/object
    // configs have no fixed per-index meaning, so those are left blank --
    // the monitor app falls back to "Ch <n>" for those.
    void ChannelNamesFromConfig(const AkChannelConfig& cfg, AkUInt32 numChannels,
        char names[][mi::scope::kMaxChannelNameLen])
    {
        for (AkUInt32 i = 0; i < numChannels && i < mi::scope::kMaxChannelsSupported; ++i)
            names[i][0] = '\0';

        if (cfg.eConfigType != AK_ChannelConfigType_Standard)
            return;

        for (const SpeakerName& s : kSpeakerNames)
        {
            if (!(cfg.uChannelMask & s.bit))
                continue;
            const AkUInt8 idx = AK::ChannelBitToIndex(s.bit, cfg.uChannelMask);
            if (idx < numChannels && idx < mi::scope::kMaxChannelsSupported)
                std::strncpy(names[idx], s.name, mi::scope::kMaxChannelNameLen - 1);
        }
    }
}

AK::IAkPlugin* CreateSignalScopeFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, SignalScopeFX());
}

AK::IAkPluginParam* CreateSignalScopeFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, SignalScopeFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(SignalScopeFX, AkPluginTypeEffect, SignalScopeConfig::CompanyID, SignalScopeConfig::PluginID)

SignalScopeFX::SignalScopeFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_iBoundScopeId(-1)
    , m_uSampleRate(0)
    , m_uNumChannels(0)
    , m_uBoundChannels(0)
    , m_bScopeBound(false)
{
    m_szBoundName[0] = '\0';
}

SignalScopeFX::~SignalScopeFX()
{
}

AKRESULT SignalScopeFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (SignalScopeFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;

    m_uSampleRate = in_rFormat.uSampleRate;
    m_uNumChannels = in_rFormat.channelConfig.uNumChannels;

    ChannelNamesFromConfig(in_rFormat.channelConfig, m_uNumChannels, m_channelNames);
    for (AkUInt32 c = 0; c < mi::scope::kMaxChannelsSupported; ++c)
        m_channelNamePtrs[c] = m_channelNames[c];

    // Force SyncScopeWriter() to (re)open the shared-memory ring on the first
    // Execute() rather than trusting whatever scope ID happened to be bound by
    // a previous Init() of a recycled plug-in instance.
    m_iBoundScopeId = -1;
    m_bScopeBound = false;

    return AK_Success;
}

AKRESULT SignalScopeFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    m_scopeWriter.Close();
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT SignalScopeFX::Reset()
{
    return AK_Success;
}

AKRESULT SignalScopeFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void SignalScopeFX::SyncScopeWriter()
{
    const AkInt32 iScopeId = m_pParams->NonRTPC.iScopeId;
    if (m_bScopeBound && iScopeId == m_iBoundScopeId &&
        strncmp(m_szBoundName, m_pParams->NonRTPC.szName, kMaxNameLen) == 0)
        return;

    m_scopeWriter.Close();
    m_bScopeBound = false;
    m_iBoundScopeId = iScopeId;
    strncpy(m_szBoundName, m_pParams->NonRTPC.szName, kMaxNameLen);
    m_szBoundName[kMaxNameLen - 1] = '\0';

    if (m_uNumChannels == 0 || iScopeId < 0)
        return;

    m_uBoundChannels = m_uNumChannels > mi::scope::kMaxChannelsSupported
        ? mi::scope::kMaxChannelsSupported
        : m_uNumChannels;

    char label[mi::scope::kMaxLabelLen];
    if (m_pParams->NonRTPC.szName[0] != '\0')
        snprintf(label, sizeof(label), "%s", m_pParams->NonRTPC.szName);
    else
        snprintf(label, sizeof(label), "Scope %d", (int)iScopeId);

    m_bScopeBound = m_scopeWriter.Open((AkUInt32)iScopeId, label, m_uSampleRate, m_uBoundChannels, m_channelNamePtrs);
}

void SignalScopeFX::Execute(AkAudioBuffer* io_pBuffer)
{
    // Pure tap: never touches the audio, only observes it. What flows through
    // this insert point is captured as-is -- put one instance before and one
    // after another effect to compare input vs output.
    if (!m_pParams->NonRTPC.bEnabled)
        return;

    SyncScopeWriter();
    if (!m_bScopeBound)
        return;

    const AkUInt32 uNumFrames = io_pBuffer->uValidFrames;
    if (uNumFrames == 0)
        return;

    const float* channelPtrs[mi::scope::kMaxChannelsSupported];
    for (AkUInt32 c = 0; c < m_uBoundChannels; ++c)
        channelPtrs[c] = (const float*)io_pBuffer->GetChannel(c);

    m_scopeWriter.Write(channelPtrs, uNumFrames);
}

AKRESULT SignalScopeFX::TimeSkip(AkUInt32 in_uFrames)
{
    return AK_DataReady;
}
