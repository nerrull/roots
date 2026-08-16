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

#include "OnsetTapFX.h"
#include "../OnsetTapConfig.h"

#include <AK/AkWwiseSDKVersion.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
    // Steady clock in nanoseconds. Only ever compared against other stamps from
    // this same writer -- the reader uses it to age events and to tell a live
    // tap from one whose sound engine stopped running.
    uint64_t NowNs()
    {
        using namespace std::chrono;
        return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    // At most this many onsets reported per block. A 512-frame block is ~11 ms,
    // and the refractory period is milliseconds at its very shortest, so this is
    // slack, not a limit anything reaches.
    const AkUInt32 kMaxHitsPerBlock = 8;
}

AK::IAkPlugin* CreateOnsetTapFX(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, OnsetTapFX());
}

AK::IAkPluginParam* CreateOnsetTapFXParams(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, OnsetTapFXParams());
}

AK_IMPLEMENT_PLUGIN_FACTORY(OnsetTapFX, AkPluginTypeEffect, OnsetTapConfig::CompanyID, OnsetTapConfig::PluginID)

OnsetTapFX::OnsetTapFX()
    : m_pParams(nullptr)
    , m_pAllocator(nullptr)
    , m_pContext(nullptr)
    , m_iBoundTapId(-1)
    , m_uSampleRate(0)
    , m_uNumChannels(0)
    , m_bBound(false)
{
    m_szBoundName[0] = '\0';
}

OnsetTapFX::~OnsetTapFX()
{
}

AKRESULT OnsetTapFX::Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat)
{
    m_pParams = (OnsetTapFXParams*)in_pParams;
    m_pAllocator = in_pAllocator;
    m_pContext = in_pContext;

    m_uSampleRate = in_rFormat.uSampleRate;
    m_uNumChannels = in_rFormat.channelConfig.uNumChannels;

    m_detector.Init(m_uSampleRate);
    SyncDetectorParams();

    // The downmix scratch is sized once, here, off the engine's maximum block:
    // Execute() runs on the audio thread and must not allocate.
    m_mono.assign(in_pContext->GlobalContext()->GetMaxBufferLength(), 0.f);

    // Force SyncWriter() to (re)open on the first Execute() rather than trusting
    // whatever a previous Init() of a recycled instance left bound.
    m_iBoundTapId = -1;
    m_bBound = false;

    return AK_Success;
}

AKRESULT OnsetTapFX::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    m_writer.Close();
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT OnsetTapFX::Reset()
{
    // Bypass, a seek, a re-use of the voice: the level history behind the
    // adaptive threshold belongs to audio that is no longer playing, and
    // carrying it over would either swallow the first hits of the new material
    // or fire on the discontinuity itself.
    m_detector.Reset();
    return AK_Success;
}

AKRESULT OnsetTapFX::GetPluginInfo(AkPluginInfo& out_rPluginInfo)
{
    out_rPluginInfo.eType = AkPluginTypeEffect;
    out_rPluginInfo.bIsInPlace = true;
    out_rPluginInfo.bCanProcessObjects = false;
    out_rPluginInfo.uBuildVersion = AK_WWISESDK_VERSION_COMBINED;
    return AK_Success;
}

void OnsetTapFX::SyncDetectorParams()
{
    mi::OnsetParams p;
    p.sensitivity = m_pParams->NonRTPC.fSensitivity;
    p.floorDb = m_pParams->NonRTPC.fFloorDb;
    p.minRiseDb = m_pParams->NonRTPC.fMinRiseDb;
    p.minIntervalMs = m_pParams->NonRTPC.fMinIntervalMs;
    p.windowMs = m_pParams->NonRTPC.fWindowMs;
    m_detector.SetParams(p);
}

void OnsetTapFX::SyncWriter()
{
    const AkInt32 iTapId = m_pParams->NonRTPC.iTapId;
    if (m_bBound && iTapId == m_iBoundTapId &&
        strncmp(m_szBoundName, m_pParams->NonRTPC.szName, kMaxNameLen) == 0)
        return;

    m_writer.Close();
    m_bBound = false;
    m_iBoundTapId = iTapId;
    strncpy(m_szBoundName, m_pParams->NonRTPC.szName, kMaxNameLen);
    m_szBoundName[kMaxNameLen - 1] = '\0';

    if (iTapId < 0)
        return;

    char label[mi::onset::kMaxLabelLen];
    if (m_pParams->NonRTPC.szName[0] != '\0')
        snprintf(label, sizeof(label), "%s", m_pParams->NonRTPC.szName);
    else
        snprintf(label, sizeof(label), "Tap %d", (int)iTapId);

    m_bBound = m_writer.Open((AkUInt32)iTapId, label, m_uSampleRate);
}

void OnsetTapFX::Execute(AkAudioBuffer* io_pBuffer)
{
    // Pure tap: the audio leaves exactly as it arrived. Everything this does is
    // observation, so an instance can be dropped on any bus without the mix
    // changing by a sample.
    if (!m_pParams->NonRTPC.bEnabled)
        return;

    SyncWriter();
    if (!m_bBound)
        return;
    SyncDetectorParams();

    const AkUInt32 uNumFrames = io_pBuffer->uValidFrames;
    if (uNumFrames == 0 || m_uNumChannels == 0)
        return;
    if (m_mono.size() < uNumFrames)
        return;   // an unexpected block size is a skipped block, never a realloc here

    // Downmix, and measure the balance while we are already touching every
    // sample. The balance is what lets a hit land where it was heard rather
    // than in the middle of the frame -- so it is measured over the block that
    // contains the transient, not over some longer average that would smear a
    // hard-panned hit back towards centre.
    const AkUInt32 uChannels = io_pBuffer->NumChannels();
    const float fScale = 1.f / (float)uChannels;
    for (AkUInt32 i = 0; i < uNumFrames; ++i)
        m_mono[i] = 0.f;

    float fLeft = 0.f, fRight = 0.f;
    for (AkUInt32 c = 0; c < uChannels; ++c)
    {
        const AkReal32* AK_RESTRICT pIn = io_pBuffer->GetChannel(c);
        if (!pIn)
            continue;
        float fEnergy = 0.f;
        for (AkUInt32 i = 0; i < uNumFrames; ++i)
        {
            m_mono[i] += pIn[i] * fScale;
            fEnergy += pIn[i] * pIn[i];
        }
        // Even channel indices sit on the left in every standard config Wwise
        // hands out (FL, FR, FC, LFE, BL, BR, ...), which is enough to place a
        // hit horizontally; a centre channel contributing to both sides would
        // only pull everything towards the middle.
        if ((c & 1) == 0)
            fLeft += fEnergy;
        else
            fRight += fEnergy;
    }

    const float fPan = (fLeft + fRight) > 1e-12f
        ? (fRight - fLeft) / (fLeft + fRight)
        : 0.f;

    mi::OnsetHit hits[kMaxHitsPerBlock];
    const AkUInt32 uHits = m_detector.Process(m_mono.data(), uNumFrames, hits, kMaxHitsPerBlock);

    const uint64_t uNow = NowNs();
    for (AkUInt32 i = 0; i < uHits; ++i)
    {
        mi::onset::Event e;
        e.hostTimeNs = uNow;
        e.strength = hits[i].strength;
        e.levelDb = hits[i].levelDb;
        e.excessDb = hits[i].excessDb;
        e.pan = fPan;
        e.channels = uChannels;
        m_writer.Push(e);
    }

    // Published every block, not only when something fires: a reader showing a
    // level next to the bar it has to clear is how you tell "nothing is playing"
    // from "the threshold is set too high", which is otherwise the same silence.
    m_writer.PublishMeter(m_detector.LevelDb(), m_detector.ThresholdDb(), uNow);
}

AKRESULT OnsetTapFX::TimeSkip(AkUInt32 in_uFrames)
{
    return AK_DataReady;
}
