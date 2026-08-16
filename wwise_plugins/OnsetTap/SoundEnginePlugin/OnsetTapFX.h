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

#ifndef OnsetTapFX_H
#define OnsetTapFX_H

#include "OnsetTapFXParams.h"
#include "../../mi_common/onset_detector.h"
#include "../../mi_common/onset_shm.h"

#include <vector>

/// Onset Tap: watches whatever passes through this insert point and publishes
/// the transients it finds to shared memory, for a program outside Wwise to
/// react to. Never alters the audio.
///
/// Why an effect and not a callback: the detection has to happen where the
/// audio is, on the audio thread, with the whole signal and no resampling in
/// between. Everything downstream only ever needs the conclusion -- "a hit, this
/// hard, panned here" -- which is a few dozen bytes per event rather than a
/// stream a reader has to keep up with. See mi_common/onset_shm.h for the
/// layout and mi_common/onset_detector.h for what counts as an onset.
class OnsetTapFX
    : public AK::IAkInPlaceEffectPlugin
{
public:
    OnsetTapFX();
    ~OnsetTapFX();

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;
    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;
    AKRESULT Reset() override;
    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;
    void Execute(AkAudioBuffer* io_pBuffer) override;
    AKRESULT TimeSkip(AkUInt32 in_uFrames) override;

private:
    // (Re)opens m_writer for the current tap ID / name, closing any previous
    // binding. Cheap to call every block: it no-ops unless something changed.
    void SyncWriter();
    // Pushes the current property values into the detector.
    void SyncDetectorParams();

    OnsetTapFXParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkEffectPluginContext* m_pContext;

    mi::OnsetDetector m_detector;
    mi::onset::TapWriter m_writer;

    AkInt32 m_iBoundTapId;
    char m_szBoundName[kMaxNameLen];
    AkUInt32 m_uSampleRate;
    AkUInt32 m_uNumChannels;
    bool m_bBound;

    // The detector is mono: a transient is an event in time, and running it per
    // channel would fire two events a millisecond apart for one stereo hit.
    // Downmixed here; the stereo balance is measured separately and travels
    // with the event as its pan.
    std::vector<float> m_mono;
};

#endif // OnsetTapFX_H
