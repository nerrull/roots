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

#include "RacineCombFX.h"
#include "../RacineCombConfig.h"

#include <AK/AkWwiseSDKVersion.h>
#include <AK/DSP/AkApplyGain.h>
#include <AK/Tools/Common/AkAssert.h>

#include <math.h>

AK::IAkPlugin* CreateRacineCombFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, RacineCombFX());
}

AK::IAkPluginParam* CreateRacineCombFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, RacineCombFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(RacineCombFX, AkPluginTypeEffect, RacineCombConfig::CompanyID, RacineCombConfig::PluginID)

RacineCombFX::RacineCombFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_pDelayMemory(nullptr)
    , m_bSettingsPrimed(false)
{
    memset(&m_previousSettings, 0, sizeof(m_previousSettings));
}

RacineCombFX::~RacineCombFX()
{
}

RacineComb::Settings RacineCombFX::CurrentSettings() const
{
    RacineComb::Settings s;
    s.fFrequency = m_pParams->RTPC.fFrequency;
    s.fFeedback = m_pParams->RTPC.fFeedback * 0.01f;
    s.fDamping = m_pParams->RTPC.fDamping * 0.01f;
    s.fWetDryMix = m_pParams->RTPC.fWetDryMix * 0.01f;
    s.fGlideMs = m_pParams->RTPC.fGlide;
    s.fOutputGain = powf(10.0f, m_pParams->RTPC.fOutputLevel * 0.05f);
    return s;
}

AKRESULT RacineCombFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (RacineCombFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;

    AkUInt32 uNumChannels = in_rFormat.channelConfig.uNumChannels;
    if (uNumChannels > RacineComb::kMaxChannels)
        uNumChannels = RacineComb::kMaxChannels;
    if (uNumChannels == 0)
        return AK_Fail;

    // Sized for the bottom of the frequency range, since Frequency is RTPC-driven
    // and can reach it at any time.
    const AkUInt32 uFramesPerChannel = RacineComb::DelayFramesPerChannel(in_rFormat.uSampleRate);
    const AkUInt32 uBytes = uFramesPerChannel * uNumChannels * sizeof(AkReal32);

    m_pDelayMemory = (AkReal32*)AK_PLUGIN_ALLOC(in_pAllocator, uBytes);
    if (!m_pDelayMemory)
        return AK_InsufficientMemory;

    m_dsp.Setup(in_rFormat.uSampleRate, uNumChannels, m_pDelayMemory);

    return Reset();
}

AKRESULT RacineCombFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    if (m_pDelayMemory)
    {
        AK_PLUGIN_FREE(in_pAllocator, m_pDelayMemory);
        m_pDelayMemory = nullptr;
    }
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT RacineCombFX::Reset()
{
    m_dsp.Reset(m_pParams ? m_pParams->RTPC.fFrequency : 220.0f);
    m_bSettingsPrimed = false;
    return AK_Success;
}

AKRESULT RacineCombFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void RacineCombFX::Execute(AkAudioBuffer* io_pBuffer)
{
    const RacineComb::Settings current = CurrentSettings();

    // On the first buffer there is nothing to ramp from, so start flat rather
    // than sweeping up from zeroed settings.
    if (!m_bSettingsPrimed)
    {
        m_previousSettings = current;
        m_bSettingsPrimed = true;
    }

    // Extends the buffer with silence while the feedback rings out. Must run
    // before reading uValidFrames, since it zero-pads to the full frame count.
    m_tailHandler.HandleTail(io_pBuffer, m_dsp.TailFrames(current));

    const AkUInt32 uNumFrames = io_pBuffer->uValidFrames;
    if (uNumFrames == 0)
    {
        m_previousSettings = current;
        return;
    }

    const AkUInt32 uNumChannels = io_pBuffer->NumChannels() < m_dsp.NumChannels()
        ? io_pBuffer->NumChannels()
        : m_dsp.NumChannels();

    for (AkUInt32 i = 0; i < uNumChannels; ++i)
    {
        AkReal32* AK_RESTRICT pBuf = (AkReal32* AK_RESTRICT)io_pBuffer->GetChannel(i);
        m_dsp.ProcessChannel(pBuf, uNumFrames, i, m_previousSettings, current);
    }

    m_previousSettings = current;
}

AKRESULT RacineCombFX::TimeSkip(AkUInt32 in_uFrames)
{
    // The delay line is not advanced while virtual: the plug-in comes back with
    // a stale tail rather than a silent one, which is cheaper and inaudible
    // after Reset on revival.
    return AK_DataReady;
}
