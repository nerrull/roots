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

#include "RacineCombFXParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

RacineCombFXParams::RacineCombFXParams()
{
}

RacineCombFXParams::~RacineCombFXParams()
{
}

RacineCombFXParams::RacineCombFXParams(const RacineCombFXParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* RacineCombFXParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, RacineCombFXParams(*this));
}

AKRESULT RacineCombFXParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        // Must match the DefaultValue entries in RacineComb.xml.
        RTPC.fFrequency = 220.0f;
        RTPC.fFeedback = 80.0f;
        RTPC.fDamping = 20.0f;
        RTPC.fWetDryMix = 100.0f;
        RTPC.fGlide = 20.0f;
        RTPC.fOutputLevel = 0.0f;
        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT RacineCombFXParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT RacineCombFXParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    // Read order must match RacineCombPlugin::GetBankParameters.
    RTPC.fFrequency = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fFeedback = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDamping = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fWetDryMix = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fGlide = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fOutputLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT RacineCombFXParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_FREQUENCY_ID:
        RTPC.fFrequency = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_FREQUENCY_ID);
        break;
    case PARAM_FEEDBACK_ID:
        RTPC.fFeedback = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_FEEDBACK_ID);
        break;
    case PARAM_DAMPING_ID:
        RTPC.fDamping = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_DAMPING_ID);
        break;
    case PARAM_WETDRYMIX_ID:
        RTPC.fWetDryMix = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_WETDRYMIX_ID);
        break;
    case PARAM_GLIDE_ID:
        RTPC.fGlide = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_GLIDE_ID);
        break;
    case PARAM_OUTPUTLEVEL_ID:
        RTPC.fOutputLevel = *((AkReal32*)in_pValue);
        m_paramChangeHandler.SetParamChange(PARAM_OUTPUTLEVEL_ID);
        break;
    default:
        eResult = AK_InvalidParameter;
        break;
    }

    return eResult;
}
