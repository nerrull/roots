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

#ifndef RacineCombFX_H
#define RacineCombFX_H

#include "RacineCombFXParams.h"
#include "RacineCombDSP.h"

#include <AK/Plugin/PluginServices/AkFXTailHandler.h>

/// See https://www.audiokinetic.com/library/edge/?source=SDK&id=soundengine__plugins__effects.html
/// for the documentation about effect plug-ins
class RacineCombFX
    : public AK::IAkInPlaceEffectPlugin
{
public:
    RacineCombFX();
    ~RacineCombFX();

    /// Plug-in initialization.
    /// Prepares the plug-in for data processing, allocates memory and sets up the initial conditions.
    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, AK::IAkEffectPluginContext* in_pContext, AK::IAkPluginParam* in_pParams, AkAudioFormat& in_rFormat) override;

    /// Release the resources upon termination of the plug-in.
    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    /// The reset action should perform any actions required to reinitialize the
    /// state of the plug-in to its original state (e.g. after Init() or on effect bypass).
    AKRESULT Reset() override;

    /// Plug-in information query mechanism used when the sound engine requires
    /// information about the plug-in to determine its behavior.
    AKRESULT GetPluginInfo(AkPluginInfo& out_rPluginInfo) override;

    /// Effect plug-in DSP execution.
    void Execute(AkAudioBuffer* io_pBuffer) override;

    /// Skips execution of some frames, when the voice is virtual playing from elapsed time.
    /// This can be used to simulate processing that would have taken place (e.g. update internal state).
    /// Return AK_DataReady or AK_NoMoreData, depending if there would be audio output or not at that point.
    AKRESULT TimeSkip(AkUInt32 in_uFrames) override;

private:
    /// Converts the current authoring-unit parameters to DSP-native units.
    RacineComb::Settings CurrentSettings() const;

    RacineCombFXParams* m_pParams;
    AK::IAkPluginMemAlloc* m_pAllocator;
    AK::IAkEffectPluginContext* m_pContext;

    RacineComb::Processor m_dsp;
    RacineComb::Settings m_previousSettings;
    AkFXTailHandler m_tailHandler;
    AkReal32* m_pDelayMemory;
    bool m_bSettingsPrimed;     ///< False until the first Execute seeds the ramp
};

#endif // RacineCombFX_H
